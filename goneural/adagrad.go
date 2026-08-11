package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// AdaGradOptimizer (Duchi et al., 2011) gives every parameter its own
// learning rate by dividing the step by the square root of the *sum* of all
// its squared gradients so far. Parameters that rarely get gradient signal
// keep taking near-full-size steps while frequently updated ones slow down,
// which suits sparse features well. The flip side of never decaying the
// accumulator is that the effective learning rate only ever shrinks, so on
// long runs progress can stall -- RMSProp exists precisely to fix that by
// swapping the sum for a decaying average.
type AdaGradOptimizer struct {
	BatchSize    int
	LearningRate float64
	Epsilon      float64

	sumSquaresW [][]float64
	sumSquaresB [][]float64
}

// NewAdaGradOptimizer creates an AdaGrad optimizer.
func NewAdaGradOptimizer(batchSize int, learningRate float64) *AdaGradOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &AdaGradOptimizer{
		BatchSize:    batchSize,
		LearningRate: learningRate,
		Epsilon:      1e-8,
	}
}

// AdaGrad returns an Optimizer that uses AdaGrad.
func AdaGrad(batchSize int, learningRate float64) Optimizer {
	return NewAdaGradOptimizer(batchSize, learningRate).Optimize
}

// Optimize implements the Optimizer signature for AdaGrad.
func (o *AdaGradOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	lenWeights := len(n.Weights)
	if o.sumSquaresW == nil {
		o.sumSquaresW = make([][]float64, lenWeights)
		o.sumSquaresB = make([][]float64, lenWeights)
		for l := 0; l < lenWeights; l++ {
			o.sumSquaresW[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
			o.sumSquaresB[l] = make([]float64, n.Biases[l].Rows)
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
			// gradient dLoss/dw, since AdaGrad scales its own descent step.
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

		for l := 0; l < lenWeights; l++ {
			rows := n.Layers[l+1].Nodes
			cols := n.Layers[l].Nodes

			wStep := adaGradStep(weightGrads[l].Flatten(), o.sumSquaresW[l], len(batch), o.LearningRate, o.Epsilon)
			bStep := adaGradStep(biasGrads[l].Flatten(), o.sumSquaresB[l], len(batch), o.LearningRate, o.Epsilon)

			n.Weights[l] = n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, wStep))
			n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, bStep))
		}
	}

	return err
}

// adaGradStep grows the lifetime sum of squared gradients in place and
// returns the per-parameter step to subtract from the weights.
func adaGradStep(gradSum, sumSquares []float64, batchLen int, lr, epsilon float64) []float64 {
	step := make([]float64, len(gradSum))
	for k, g := range gradSum {
		g /= float64(batchLen)
		sumSquares[k] += g * g
		step[k] = lr * g / (math.Sqrt(sumSquares[k]) + epsilon)
	}
	return step
}
