package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// AdamaxOptimizer is the Adamax variant from the original Adam paper
// (Kingma & Ba, 2014): the second moment's L2 norm is replaced with an
// exponentially decaying *infinity* norm -- each parameter divides by the
// largest recent |gradient| it has seen (decayed by Beta2 per step) rather
// than by an average of squares. That makes the step size insensitive to
// occasional huge gradients (they set the denominator instead of being
// squared into it) and removes the need for second-moment bias correction,
// since the max is not pulled toward zero the way an average is.
type AdamaxOptimizer struct {
	BatchSize    int
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64

	mW [][]float64
	uW [][]float64
	mB [][]float64
	uB [][]float64
	t  int
}

// NewAdamaxOptimizer creates an Adamax optimizer with the paper's defaults
// (beta1=0.9, beta2=0.999, epsilon=1e-8).
func NewAdamaxOptimizer(batchSize int, learningRate float64) *AdamaxOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &AdamaxOptimizer{
		BatchSize:    batchSize,
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
	}
}

// Adamax returns an Optimizer that uses the infinity-norm variant of Adam.
func Adamax(batchSize int, learningRate float64) Optimizer {
	return NewAdamaxOptimizer(batchSize, learningRate).Optimize
}

// Optimize implements the Optimizer signature.
func (o *AdamaxOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	lenWeights := len(n.Weights)
	if o.mW == nil {
		o.mW = make([][]float64, lenWeights)
		o.uW = make([][]float64, lenWeights)
		o.mB = make([][]float64, lenWeights)
		o.uB = make([][]float64, lenWeights)
		for l := 0; l < lenWeights; l++ {
			o.mW[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
			o.uW[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
			o.mB[l] = make([]float64, n.Biases[l].Rows)
			o.uB[l] = make([]float64, n.Biases[l].Rows)
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

		for l := 0; l < lenWeights; l++ {
			rows := n.Layers[l+1].Nodes
			cols := n.Layers[l].Nodes

			wStep := adamaxStep(weightGrads[l].Flatten(), o.mW[l], o.uW[l], len(batch), o.LearningRate, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1)
			bStep := adamaxStep(biasGrads[l].Flatten(), o.mB[l], o.uB[l], len(batch), o.LearningRate, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1)

			n.Weights[l] = n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, wStep))
			n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, bStep))
		}
	}

	return err
}

// adamaxStep updates the first moment and the decayed infinity norm u in
// place and returns the per-parameter step. Only the first moment needs
// bias correction; u starts from whatever |gradient| arrives first.
func adamaxStep(gradSum, m, u []float64, batchLen int, lr, beta1, beta2, epsilon, biasCorrection1 float64) []float64 {
	step := make([]float64, len(gradSum))
	for k, g := range gradSum {
		g /= float64(batchLen)
		m[k] = beta1*m[k] + (1-beta1)*g
		u[k] = math.Max(beta2*u[k], math.Abs(g))
		mHat := m[k] / biasCorrection1
		step[k] = lr * mHat / (u[k] + epsilon)
	}
	return step
}
