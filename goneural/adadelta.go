package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// AdaDeltaOptimizer (Zeiler, 2012) is RMSProp taken one step further: it
// needs no learning rate at all. Each parameter divides a decaying RMS of
// its past *updates* by a decaying RMS of its past *gradients*, so the step
// size is derived entirely from the ratio of how far the parameter has been
// moving to how loud its gradients are. Steps start tiny (on the order of
// sqrt(epsilon)) and grow as update history accumulates, so AdaDelta tends
// to need more epochs to get going than Adam or RMSProp.
type AdaDeltaOptimizer struct {
	BatchSize int
	Rho       float64
	Epsilon   float64

	gradSquaresW [][]float64
	stepSquaresW [][]float64
	gradSquaresB [][]float64
	stepSquaresB [][]float64
}

// NewAdaDeltaOptimizer creates an AdaDelta optimizer with the paper's
// defaults (rho=0.95, epsilon=1e-6).
func NewAdaDeltaOptimizer(batchSize int) *AdaDeltaOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &AdaDeltaOptimizer{
		BatchSize: batchSize,
		Rho:       0.95,
		Epsilon:   1e-6,
	}
}

// AdaDelta returns an Optimizer that uses AdaDelta. Note the absence of a
// learning-rate parameter -- deriving the step size from accumulated state
// is the method's point.
func AdaDelta(batchSize int) Optimizer {
	return NewAdaDeltaOptimizer(batchSize).Optimize
}

// Optimize implements the Optimizer signature.
func (o *AdaDeltaOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	lenWeights := len(n.Weights)
	if o.gradSquaresW == nil {
		o.gradSquaresW = make([][]float64, lenWeights)
		o.stepSquaresW = make([][]float64, lenWeights)
		o.gradSquaresB = make([][]float64, lenWeights)
		o.stepSquaresB = make([][]float64, lenWeights)
		for l := 0; l < lenWeights; l++ {
			sizeW := n.Weights[l].Rows * n.Weights[l].Columns
			o.gradSquaresW[l] = make([]float64, sizeW)
			o.stepSquaresW[l] = make([]float64, sizeW)
			o.gradSquaresB[l] = make([]float64, n.Biases[l].Rows)
			o.stepSquaresB[l] = make([]float64, n.Biases[l].Rows)
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

			wStep := adaDeltaStep(weightGrads[l].Flatten(), o.gradSquaresW[l], o.stepSquaresW[l], len(batch), o.Rho, o.Epsilon)
			bStep := adaDeltaStep(biasGrads[l].Flatten(), o.gradSquaresB[l], o.stepSquaresB[l], len(batch), o.Rho, o.Epsilon)

			n.Weights[l] = n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, wStep))
			n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, bStep))
		}
	}

	return err
}

// adaDeltaStep updates the decaying squared-gradient and squared-step
// accumulators in place and returns the per-parameter step: the gradient
// rescaled by RMS(past steps) / RMS(current gradients).
func adaDeltaStep(gradSum, gradSquares, stepSquares []float64, batchLen int, rho, epsilon float64) []float64 {
	step := make([]float64, len(gradSum))
	for k, g := range gradSum {
		g /= float64(batchLen)
		gradSquares[k] = rho*gradSquares[k] + (1-rho)*g*g
		s := math.Sqrt(stepSquares[k]+epsilon) / math.Sqrt(gradSquares[k]+epsilon) * g
		stepSquares[k] = rho*stepSquares[k] + (1-rho)*s*s
		step[k] = s
	}
	return step
}
