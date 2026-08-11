package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// AMSGradOptimizer is the AMSGrad variant of Adam (Reddi et al., 2018).
// Adam divides by a *decaying* average of squared gradients, so a burst of
// large gradients is eventually forgotten and the effective step size can
// grow back -- the behaviour behind Adam's known non-convergence examples.
// AMSGrad instead divides by the elementwise high-water mark of that
// average: once a parameter has seen large gradients, its step size never
// re-inflates, which restores a convergence guarantee at the cost of a
// permanently more conservative step.
type AMSGradOptimizer struct {
	BatchSize    int
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64

	mW    [][]float64
	vW    [][]float64
	vMaxW [][]float64
	mB    [][]float64
	vB    [][]float64
	vMaxB [][]float64
	t     int
}

// NewAMSGradOptimizer creates an AMSGrad optimizer with the standard Adam
// defaults (beta1=0.9, beta2=0.999, epsilon=1e-8).
func NewAMSGradOptimizer(batchSize int, learningRate float64) *AMSGradOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &AMSGradOptimizer{
		BatchSize:    batchSize,
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
	}
}

// AMSGrad returns an Optimizer that uses the AMSGrad variant of Adam.
func AMSGrad(batchSize int, learningRate float64) Optimizer {
	return NewAMSGradOptimizer(batchSize, learningRate).Optimize
}

// Optimize implements the Optimizer signature.
func (o *AMSGradOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	lenWeights := len(n.Weights)
	if o.mW == nil {
		o.mW = make([][]float64, lenWeights)
		o.vW = make([][]float64, lenWeights)
		o.vMaxW = make([][]float64, lenWeights)
		o.mB = make([][]float64, lenWeights)
		o.vB = make([][]float64, lenWeights)
		o.vMaxB = make([][]float64, lenWeights)
		for l := 0; l < lenWeights; l++ {
			sizeW := n.Weights[l].Rows * n.Weights[l].Columns
			o.mW[l] = make([]float64, sizeW)
			o.vW[l] = make([]float64, sizeW)
			o.vMaxW[l] = make([]float64, sizeW)
			o.mB[l] = make([]float64, n.Biases[l].Rows)
			o.vB[l] = make([]float64, n.Biases[l].Rows)
			o.vMaxB[l] = make([]float64, n.Biases[l].Rows)
		}
	}

	err := 0.0

	for i := 0; i < len(dataSet); i += o.BatchSize {
		batch := dataSet.Batch(i, o.BatchSize)
		if len(batch) == 0 {
			continue
		}

		weightGrads := make([]matrix.Matrix, lenWeights)
		biasGrads := make([]matrix.Matrix, lenWeights)
		for l := 0; l < lenWeights; l++ {
			weightGrads[l] = matrix.New(n.Weights[l].Rows, n.Weights[l].Columns, nil)
			biasGrads[l] = matrix.New(n.Biases[l].Rows, n.Biases[l].Columns, nil)
		}

		for _, ds := range batch {
			outputs := n.predict(matrix.NewFromArray(ds.Inputs))
			targets := matrix.NewFromArray(ds.Targets)
			err += n.Loss.F(outputs, targets)

			// Same backprop recurrence as MBGD, but with the true (positive)
			// gradient dLoss/dw, since AMSGrad derives its own step.
			delta := outputDelta(n, outputs, targets).Scale(-1)
			for l := lenWeights - 1; l >= 0; l-- {
				weightGrads[l] = weightGrads[l].AddMatrix(delta.Multiply(n.Activations[l].Transpose()))
				biasGrads[l] = biasGrads[l].AddMatrix(delta)

				if l > 0 {
					delta = n.Weights[l].Transpose().Multiply(delta)
					delta = n.Activations[l].
						Map(func(val float64, x, y int) float64 {
							return n.Layers[l].Activator.FPrime(val)
						}).
						HadamardProduct(delta)
				}
			}
		}

		o.t++
		biasCorrection1 := 1 - math.Pow(o.Beta1, float64(o.t))
		biasCorrection2 := 1 - math.Pow(o.Beta2, float64(o.t))

		for l := 0; l < lenWeights; l++ {
			rows := n.Layers[l+1].Nodes
			cols := n.Layers[l].Nodes

			wStep := amsGradStep(weightGrads[l].Flatten(), o.mW[l], o.vW[l], o.vMaxW[l], len(batch), o.LearningRate, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1, biasCorrection2)
			bStep := amsGradStep(biasGrads[l].Flatten(), o.mB[l], o.vB[l], o.vMaxB[l], len(batch), o.LearningRate, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1, biasCorrection2)

			n.Weights[l] = n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, wStep))
			n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, bStep))
		}
	}

	return err
}

// amsGradStep updates m, v and the high-water mark vMax in place and
// returns the per-parameter step. The denominator uses vMax rather than v,
// which is the entire difference from adamStep.
func amsGradStep(gradSum, m, v, vMax []float64, batchLen int, lr, beta1, beta2, epsilon, biasCorrection1, biasCorrection2 float64) []float64 {
	step := make([]float64, len(gradSum))
	for k, g := range gradSum {
		g /= float64(batchLen)
		m[k] = beta1*m[k] + (1-beta1)*g
		v[k] = beta2*v[k] + (1-beta2)*g*g
		if v[k] > vMax[k] {
			vMax[k] = v[k]
		}
		mHat := m[k] / biasCorrection1
		vHat := vMax[k] / biasCorrection2
		step[k] = lr * mHat / (math.Sqrt(vHat) + epsilon)
	}
	return step
}
