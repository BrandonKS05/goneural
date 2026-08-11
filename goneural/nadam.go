package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// NadamOptimizer is Adam with Nesterov momentum folded in (Dozat, 2016).
// Where Adam steps along the bias-corrected running mean of past gradients,
// Nadam adds one more decayed helping of the *current* gradient on top --
// the same look-ahead trick NesterovSGD applies to classic momentum -- so
// the step anticipates where the first moment is headed. In practice it
// behaves like Adam with slightly snappier reactions to gradient changes.
type NadamOptimizer struct {
	BatchSize    int
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64

	mW [][]float64
	vW [][]float64
	mB [][]float64
	vB [][]float64
	t  int
}

// NewNadamOptimizer creates a Nadam optimizer with the standard Adam
// defaults (beta1=0.9, beta2=0.999, epsilon=1e-8).
func NewNadamOptimizer(batchSize int, learningRate float64) *NadamOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &NadamOptimizer{
		BatchSize:    batchSize,
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
	}
}

// Nadam returns an Optimizer that uses Nesterov-accelerated Adam.
func Nadam(batchSize int, learningRate float64) Optimizer {
	return NewNadamOptimizer(batchSize, learningRate).Optimize
}

// Optimize implements the Optimizer signature.
func (o *NadamOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
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
			// gradient dLoss/dw, since Nadam derives its own step.
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

			wStep := nadamStep(weightGrads[l].Flatten(), o.mW[l], o.vW[l], len(batch), o.LearningRate, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1, biasCorrection2)
			bStep := nadamStep(biasGrads[l].Flatten(), o.mB[l], o.vB[l], len(batch), o.LearningRate, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1, biasCorrection2)

			n.Weights[l] = n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, wStep))
			n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, bStep))
		}
	}

	return err
}

// nadamStep updates the moment estimates m/v in place and returns the
// per-parameter step. It differs from adamStep only in the numerator: the
// bias-corrected momentum is decayed by beta1 once more and topped up with
// the current gradient's bias-corrected share (the Nesterov look-ahead).
func nadamStep(gradSum, m, v []float64, batchLen int, lr, beta1, beta2, epsilon, biasCorrection1, biasCorrection2 float64) []float64 {
	step := make([]float64, len(gradSum))
	for k, g := range gradSum {
		g /= float64(batchLen)
		m[k] = beta1*m[k] + (1-beta1)*g
		v[k] = beta2*v[k] + (1-beta2)*g*g
		mHat := m[k] / biasCorrection1
		vHat := v[k] / biasCorrection2
		mNesterov := beta1*mHat + (1-beta1)*g/biasCorrection1
		step[k] = lr * mNesterov / (math.Sqrt(vHat) + epsilon)
	}
	return step
}
