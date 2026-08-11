package goneural

import (
	"github.com/BrandonKS05/goneural/matrix"
)

// LionOptimizer implements Lion (EvoLved Sign Momentum, Chen et al., 2023),
// an optimizer found by symbolic program search rather than derivation.
// Each update moves every parameter by exactly +-LearningRate: the step is
// the *sign* of an interpolation between the momentum and the current
// gradient, so the gradient's magnitude only ever influences direction
// through the momentum it leaves behind. That uniform step makes Lion
// memory-light (one moment buffer instead of Adam's two) and typically
// calls for a learning rate 3-10x smaller than Adam's, since every
// parameter moves the full step every time.
type LionOptimizer struct {
	BatchSize    int
	LearningRate float64
	Beta1        float64
	Beta2        float64

	// WeightDecay, when positive, applies decoupled weight decay exactly as
	// AdamW does: weights shrink by LearningRate*WeightDecay*w each update,
	// outside the sign-based step.
	WeightDecay float64

	mW [][]float64
	mB [][]float64
}

// NewLionOptimizer creates a Lion optimizer with the paper's defaults
// (beta1=0.9, beta2=0.99) and no weight decay.
func NewLionOptimizer(batchSize int, learningRate float64) *LionOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &LionOptimizer{
		BatchSize:    batchSize,
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.99,
	}
}

// Lion returns an Optimizer that uses the Lion sign-momentum method.
func Lion(batchSize int, learningRate float64) Optimizer {
	return NewLionOptimizer(batchSize, learningRate).Optimize
}

// Optimize implements the Optimizer signature.
func (o *LionOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	lenWeights := len(n.Weights)
	if o.mW == nil {
		o.mW = make([][]float64, lenWeights)
		o.mB = make([][]float64, lenWeights)
		for l := 0; l < lenWeights; l++ {
			o.mW[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
			o.mB[l] = make([]float64, n.Biases[l].Rows)
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

		for l := 0; l < lenWeights; l++ {
			rows := n.Layers[l+1].Nodes
			cols := n.Layers[l].Nodes

			wStep := lionStep(weightGrads[l].Flatten(), o.mW[l], len(batch), o.LearningRate, o.Beta1, o.Beta2)
			bStep := lionStep(biasGrads[l].Flatten(), o.mB[l], len(batch), o.LearningRate, o.Beta1, o.Beta2)

			newW := n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, wStep))
			if o.WeightDecay > 0 {
				newW = newW.SubtractMatrix(n.Weights[l].Scale(o.LearningRate * o.WeightDecay))
			}

			n.Weights[l] = newW
			n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, bStep))
		}
	}

	return err
}

// lionStep updates the momentum in place and returns the per-parameter
// step. The interpolation feeding sign() uses Beta1 while the momentum
// retained for later steps decays with Beta2 -- the two-timescale split is
// what the paper's search found to matter.
func lionStep(gradSum, m []float64, batchLen int, lr, beta1, beta2 float64) []float64 {
	step := make([]float64, len(gradSum))
	for k, g := range gradSum {
		g /= float64(batchLen)
		c := beta1*m[k] + (1-beta1)*g
		switch {
		case c > 0:
			step[k] = lr
		case c < 0:
			step[k] = -lr
		}
		m[k] = beta2*m[k] + (1-beta2)*g
	}
	return step
}
