package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// RAdamOptimizer is Rectified Adam (Liu et al., 2019).
//
// Adam's per-parameter denominator is an average of only a handful of squared
// gradients during the first few steps, so its variance is enormous and the
// effective step size swings wildly -- which is why Adam in practice needs a
// hand-tuned warmup to avoid diving into a bad region before it has any idea
// of the curvature. RAdam replaces that warmup with a closed form. It tracks
// rho, the length of the simple moving average that the second moment is
// implicitly estimating, and while rho is too short for the variance to even
// be defined (rho <= 4) it falls back to an un-adapted momentum step. Once the
// estimate becomes usable it scales the adaptive step by a rectification term
// that cancels the remaining variance. That term rises toward 1 as training
// proceeds, so late training is ordinary Adam and the warmup disappears on its
// own rather than on a schedule someone had to guess.
type RAdamOptimizer struct {
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

// NewRAdamOptimizer creates a rectified Adam optimizer with the paper's
// defaults (beta1=0.9, beta2=0.999, epsilon=1e-8).
func NewRAdamOptimizer(batchSize int, learningRate float64) *RAdamOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &RAdamOptimizer{
		BatchSize:    batchSize,
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
	}
}

// RAdam returns an Optimizer that uses Adam with variance rectification.
func RAdam(batchSize int, learningRate float64) Optimizer {
	return NewRAdamOptimizer(batchSize, learningRate).Optimize
}

// Optimize implements the Optimizer signature.
func (o *RAdamOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
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
		rect, rectify := radamRectification(radamRho(o.Beta2, o.t))

		for l := 0; l < lenWeights; l++ {
			rows := n.Layers[l+1].Nodes
			cols := n.Layers[l].Nodes

			wStep := radamStep(weightGrads[l].Flatten(), o.mW[l], o.vW[l], len(batch), o.LearningRate, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1, biasCorrection2, rect, rectify)
			bStep := radamStep(biasGrads[l].Flatten(), o.mB[l], o.vB[l], len(batch), o.LearningRate, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1, biasCorrection2, rect, rectify)

			n.Weights[l] = n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, wStep))
			n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, bStep))
		}
	}

	return err
}

// radamRho returns the maximum length of the simple moving average that the
// second moment approximates (rhoInf, reached only in the limit) and its
// length at step t. rhoT climbs from 1 toward rhoInf, so it doubles as the
// "how much evidence do we have yet" counter the rectification keys off.
func radamRho(beta2 float64, t int) (rhoInf, rhoT float64) {
	rhoInf = 2/(1-beta2) - 1
	beta2T := math.Pow(beta2, float64(t))
	rhoT = rhoInf - 2*float64(t)*beta2T/(1-beta2T)
	return rhoInf, rhoT
}

// radamRectification returns the variance rectification term r_t along with
// whether the second moment is trustworthy enough to divide by at all. The
// variance of the adaptive term is only finite once rho exceeds 4, so below
// that the caller must skip the denominator entirely rather than scale it.
func radamRectification(rhoInf, rhoT float64) (float64, bool) {
	if rhoT <= 4 {
		return 0, false
	}

	numerator := (rhoT - 4) * (rhoT - 2) * rhoInf
	denominator := (rhoInf - 4) * (rhoInf - 2) * rhoT
	return math.Sqrt(numerator / denominator), true
}

// radamStep updates both moments in place and returns the per-parameter step.
// When rectify is false the second moment is ignored completely and this
// degenerates to bias-corrected momentum SGD -- that is the whole point of the
// early phase, not a fallback for a degenerate case.
func radamStep(gradSum, m, v []float64, batchLen int, lr, beta1, beta2, epsilon, biasCorrection1, biasCorrection2, rect float64, rectify bool) []float64 {
	step := make([]float64, len(gradSum))
	for k, g := range gradSum {
		g /= float64(batchLen)
		m[k] = beta1*m[k] + (1-beta1)*g
		v[k] = beta2*v[k] + (1-beta2)*g*g
		mHat := m[k] / biasCorrection1

		if !rectify {
			step[k] = lr * mHat
			continue
		}

		vHat := math.Sqrt(v[k] / biasCorrection2)
		step[k] = lr * mHat * rect / (vHat + epsilon)
	}
	return step
}
