package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// AdaBeliefOptimizer is AdaBelief (Zhuang et al., 2020).
//
// The change from Adam is one term: the second moment accumulates the squared
// distance between the gradient and the momentum, (g - m)^2, instead of the
// squared gradient g^2. Adam asks "how big are the gradients here?"; AdaBelief
// asks "how much does the gradient disagree with what I predicted?" -- the
// momentum is the prediction and the second moment is the running surprise.
//
// The consequence is the opposite behaviour on a long, consistent slope. There
// Adam sees large g^2, inflates its denominator, and takes small steps down a
// gradient it should trust. AdaBelief sees g land right where m said it would,
// so the surprise is near zero, the denominator collapses, and the step grows.
// On a noisy, oscillating surface the two swap: the surprise is large, the
// denominator inflates, and AdaBelief becomes the cautious one.
//
// Epsilon is added inside the accumulator as well as at the division, exactly
// as the paper specifies. That is not the usual cosmetic guard against a zero
// denominator: without it, a perfectly predicted gradient drives the running
// surprise to zero and the step to infinity.
type AdaBeliefOptimizer struct {
	BatchSize    int
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64

	mW [][]float64
	sW [][]float64
	mB [][]float64
	sB [][]float64
	t  int
}

// NewAdaBeliefOptimizer creates an AdaBelief optimizer with the paper's
// defaults (beta1=0.9, beta2=0.999, epsilon=1e-8).
func NewAdaBeliefOptimizer(batchSize int, learningRate float64) *AdaBeliefOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &AdaBeliefOptimizer{
		BatchSize:    batchSize,
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
	}
}

// AdaBelief returns an Optimizer that scales steps by the running surprise in
// the gradient rather than its running magnitude.
func AdaBelief(batchSize int, learningRate float64) Optimizer {
	return NewAdaBeliefOptimizer(batchSize, learningRate).Optimize
}

// Optimize implements the Optimizer signature.
func (o *AdaBeliefOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	lenWeights := len(n.Weights)
	if o.mW == nil {
		o.mW = make([][]float64, lenWeights)
		o.sW = make([][]float64, lenWeights)
		o.mB = make([][]float64, lenWeights)
		o.sB = make([][]float64, lenWeights)
		for l := 0; l < lenWeights; l++ {
			o.mW[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
			o.sW[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
			o.mB[l] = make([]float64, n.Biases[l].Rows)
			o.sB[l] = make([]float64, n.Biases[l].Rows)
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

			wStep := adaBeliefStep(weightGrads[l].Flatten(), o.mW[l], o.sW[l], len(batch), o.LearningRate, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1, biasCorrection2)
			bStep := adaBeliefStep(biasGrads[l].Flatten(), o.mB[l], o.sB[l], len(batch), o.LearningRate, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1, biasCorrection2)

			n.Weights[l] = n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, wStep))
			n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, bStep))
		}
	}

	return err
}

// adaBeliefStep updates the momentum and the running surprise in place and
// returns the per-parameter step. Note s accumulates (g - m) *after* m has
// been updated with this step's gradient, which is what makes the residual a
// one-step-ahead prediction error rather than a stale one.
func adaBeliefStep(gradSum, m, s []float64, batchLen int, lr, beta1, beta2, epsilon, biasCorrection1, biasCorrection2 float64) []float64 {
	step := make([]float64, len(gradSum))
	for k, g := range gradSum {
		g /= float64(batchLen)
		m[k] = beta1*m[k] + (1-beta1)*g

		surprise := g - m[k]
		s[k] = beta2*s[k] + (1-beta2)*surprise*surprise + epsilon

		mHat := m[k] / biasCorrection1
		sHat := s[k] / biasCorrection2
		step[k] = lr * mHat / (math.Sqrt(sHat) + epsilon)
	}
	return step
}
