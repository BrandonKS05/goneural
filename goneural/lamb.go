package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// LAMBOptimizer is LAMB, Layer-wise Adaptive Moments (You et al., 2019).
//
// Every optimizer in this package so far picks a step size per *parameter*.
// LAMB adds a second, coarser decision on top: a step size per *layer*. It
// computes an ordinary Adam direction, then rescales that whole layer's update
// by the trust ratio ||w|| / ||u|| -- the size of the layer's weights divided
// by the size of the update about to be applied to them. The applied step is
// therefore a fixed *fraction* of each layer's own scale, so a layer whose
// weights are tiny does not get shoved around by an update sized for a layer
// whose weights are large.
//
// That matters because layers in the same network routinely differ in weight
// norm by an order of magnitude, and a single global learning rate is a
// compromise between them. Removing that compromise is what let LAMB scale
// BERT pre-training to a 32k batch: with the per-layer ratio doing the
// normalizing, the global learning rate stops being the thing that has to be
// re-tuned every time the batch size changes.
//
// The trust ratio is bypassed (set to 1) when either norm is zero, which is
// the paper's convention -- a zero-weight layer has no scale to be relative
// to, and a zero update needs no rescaling.
type LAMBOptimizer struct {
	BatchSize    int
	LearningRate float64
	WeightDecay  float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64

	mW [][]float64
	vW [][]float64
	mB [][]float64
	vB [][]float64
	t  int
}

// NewLAMBOptimizer creates a LAMB optimizer with the paper's defaults
// (beta1=0.9, beta2=0.999, epsilon=1e-6).
func NewLAMBOptimizer(batchSize int, learningRate, weightDecay float64) *LAMBOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &LAMBOptimizer{
		BatchSize:    batchSize,
		LearningRate: learningRate,
		WeightDecay:  weightDecay,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-6,
	}
}

// LAMB returns an Optimizer that applies Adam directions rescaled by a
// per-layer trust ratio.
func LAMB(batchSize int, learningRate, weightDecay float64) Optimizer {
	return NewLAMBOptimizer(batchSize, learningRate, weightDecay).Optimize
}

// Optimize implements the Optimizer signature.
func (o *LAMBOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	lenWeights := len(n.Weights)
	if o.mW == nil {
		o.mW = make([][]float64, lenWeights)
		o.vW = make([][]float64, lenWeights)
		o.mB = make([][]float64, lenWeights)
		o.vB = make([][]float64, lenWeights)
		for l := 0; l < lenWeights; l++ {
			o.mW[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
			o.vW[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
			o.mB[l] = make([]float64, n.Biases[l].Rows)
			o.vB[l] = make([]float64, n.Biases[l].Rows)
		}
	}

	err := 0.0

	for i := 0; i < len(dataSet); i += o.BatchSize {
		batch := dataSet.Batch(i, o.BatchSize)
		if len(batch) == 0 {
			continue
		}

		weightGrads, biasGrads, batchErr := accumulateBatchGradients(n, batch)
		err += batchErr

		o.t++
		biasCorrection1 := 1 - math.Pow(o.Beta1, float64(o.t))
		biasCorrection2 := 1 - math.Pow(o.Beta2, float64(o.t))

		for l := 0; l < lenWeights; l++ {
			rows := n.Layers[l+1].Nodes
			cols := n.Layers[l].Nodes

			// The bias vector is its own "layer" for trust purposes: it has a
			// scale of its own and no reason to inherit the weight matrix's.
			wUpdate := lambDirection(weightGrads[l].Flatten(), o.mW[l], o.vW[l], n.Weights[l].Flatten(), len(batch), o.WeightDecay, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1, biasCorrection2)
			bUpdate := lambDirection(biasGrads[l].Flatten(), o.mB[l], o.vB[l], n.Biases[l].Flatten(), len(batch), o.WeightDecay, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1, biasCorrection2)

			wTrust := lambTrustRatio(n.Weights[l].Norm(), matrix.Unflatten(rows, cols, wUpdate).Norm())
			bTrust := lambTrustRatio(n.Biases[l].Norm(), matrix.Unflatten(rows, 1, bUpdate).Norm())

			for k := range wUpdate {
				wUpdate[k] *= o.LearningRate * wTrust
			}
			for k := range bUpdate {
				bUpdate[k] *= o.LearningRate * bTrust
			}

			n.Weights[l] = n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, wUpdate))
			n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, bUpdate))
		}
	}

	return err
}

// lambDirection updates both moments in place and returns the unscaled update
// direction for one layer: the bias-corrected Adam ratio plus decoupled weight
// decay. The learning rate is deliberately left out -- the caller needs the
// norm of this raw direction to form the trust ratio, and folding the learning
// rate in first would cancel out of the ratio anyway.
func lambDirection(gradSum, m, v, params []float64, batchLen int, weightDecay, beta1, beta2, epsilon, biasCorrection1, biasCorrection2 float64) []float64 {
	update := make([]float64, len(gradSum))
	for k, g := range gradSum {
		g /= float64(batchLen)
		m[k] = beta1*m[k] + (1-beta1)*g
		v[k] = beta2*v[k] + (1-beta2)*g*g

		mHat := m[k] / biasCorrection1
		vHat := v[k] / biasCorrection2
		update[k] = mHat/(math.Sqrt(vHat)+epsilon) + weightDecay*params[k]
	}
	return update
}

// lambTrustRatio returns ||w|| / ||u||, or 1 when either norm vanishes and the
// ratio would be undefined or meaningless.
func lambTrustRatio(paramNorm, updateNorm float64) float64 {
	if paramNorm == 0 || updateNorm == 0 {
		return 1
	}
	return paramNorm / updateNorm
}
