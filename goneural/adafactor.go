package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// AdafactorOptimizer is Adafactor (Shazeer & Stern, 2018).
//
// Every Adam-family optimizer here keeps a second moment the same size as the
// parameters, which doubles or triples the memory a model costs to train.
// Adafactor's observation is that for a weight *matrix* that second moment is
// close to rank one in practice, so it need not be stored in full: keeping one
// running sum per row and one per column is enough to reconstruct a good
// approximation on demand as the outer product R C / sum(R). A rows x cols
// matrix therefore costs rows + cols numbers of state instead of rows * cols.
//
// This implementation factors the weight matrices and stores bias vectors in
// full, which is the standard rule -- a vector has no second dimension to
// factor along, and rows + 1 would save nothing.
//
// Two further pieces of the paper come along with the factoring, because
// without them it does not train stably:
//
//   - Decay that grows with time (beta2_t = 1 - t^-DecayPower) instead of a
//     constant beta2. At t=1 the accumulator is exactly the current gradient,
//     so there is no zero-initialization bias to correct and no bias-correction
//     term anywhere in this optimizer.
//   - Update clipping by RMS rather than by norm. The update is divided by
//     max(1, RMS(u)/ClipThreshold), which caps the *typical* element of the
//     update instead of its total length, so the cap does not tighten simply
//     because a layer has more parameters in it.
type AdafactorOptimizer struct {
	BatchSize     int
	LearningRate  float64
	DecayPower    float64
	ClipThreshold float64
	Epsilon       float64

	rowW [][]float64
	colW [][]float64
	vB   [][]float64
	t    int
}

// NewAdafactorOptimizer creates an Adafactor optimizer with the paper's
// defaults (decay power 0.8, clip threshold 1.0, epsilon 1e-30).
func NewAdafactorOptimizer(batchSize int, learningRate float64) *AdafactorOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &AdafactorOptimizer{
		BatchSize:     batchSize,
		LearningRate:  learningRate,
		DecayPower:    0.8,
		ClipThreshold: 1,
		Epsilon:       1e-30,
	}
}

// Adafactor returns an Optimizer that stores the second moment of each weight
// matrix as a rank-one row/column factorization.
func Adafactor(batchSize int, learningRate float64) Optimizer {
	return NewAdafactorOptimizer(batchSize, learningRate).Optimize
}

// Optimize implements the Optimizer signature.
func (o *AdafactorOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	lenWeights := len(n.Weights)
	if o.rowW == nil {
		o.rowW = make([][]float64, lenWeights)
		o.colW = make([][]float64, lenWeights)
		o.vB = make([][]float64, lenWeights)
		for l := 0; l < lenWeights; l++ {
			o.rowW[l] = make([]float64, n.Weights[l].Rows)
			o.colW[l] = make([]float64, n.Weights[l].Columns)
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
		beta2t := adafactorDecay(o.t, o.DecayPower)

		for l := 0; l < lenWeights; l++ {
			rows := n.Layers[l+1].Nodes
			cols := n.Layers[l].Nodes

			wUpdate := adafactorFactoredUpdate(weightGrads[l], o.rowW[l], o.colW[l], len(batch), beta2t, o.Epsilon, o.ClipThreshold)
			bUpdate := adafactorVectorUpdate(biasGrads[l].Flatten(), o.vB[l], len(batch), beta2t, o.Epsilon, o.ClipThreshold)

			n.Weights[l] = n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, wUpdate).Scale(o.LearningRate))
			n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, bUpdate).Scale(o.LearningRate))
		}
	}

	return err
}

// adafactorDecay returns beta2 at step t. It is 0 at t=1 -- the accumulator
// starts as the raw squared gradient rather than as a decayed average of a
// zero it was initialized to -- and approaches 1 as training goes on, so the
// running estimate grows steadier the more evidence it has.
func adafactorDecay(t int, decayPower float64) float64 {
	return 1 - math.Pow(float64(t), -decayPower)
}

// adafactorFactoredUpdate updates the row and column accumulators in place and
// returns the RMS-clipped update for a weight matrix. The full second moment
// is never materialized as state: it is rebuilt each step as the outer product
// of the two accumulators, normalized by their shared total so the
// reconstruction has the right scale.
func adafactorFactoredUpdate(grad matrix.Matrix, rowAcc, colAcc []float64, batchLen int, beta2t, epsilon, clipThreshold float64) []float64 {
	rows, cols := grad.Rows, grad.Columns

	g := grad.Divide(float64(batchLen))
	squares := g.Map(func(val float64, x, y int) float64 {
		return val*val + epsilon
	})

	rowSums := squares.RowSums().Flatten()
	colSums := squares.ColumnSums().Flatten()

	for i := range rowAcc {
		rowAcc[i] = beta2t*rowAcc[i] + (1-beta2t)*rowSums[i]
	}
	for j := range colAcc {
		colAcc[j] = beta2t*colAcc[j] + (1-beta2t)*colSums[j]
	}

	total := 0.0
	for _, v := range rowAcc {
		total += v
	}

	gFlat := g.Flatten()
	update := make([]float64, len(gFlat))
	if total == 0 {
		return update
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			estimate := rowAcc[i] * colAcc[j] / total
			update[i*cols+j] = gFlat[i*cols+j] / math.Sqrt(estimate)
		}
	}

	return adafactorClip(update, clipThreshold)
}

// adafactorVectorUpdate is the unfactored path used for bias vectors: an
// ordinary running second moment, sharing the same time-varying decay and RMS
// clipping as the factored path so the two move in step.
func adafactorVectorUpdate(gradSum, v []float64, batchLen int, beta2t, epsilon, clipThreshold float64) []float64 {
	update := make([]float64, len(gradSum))
	for k, g := range gradSum {
		g /= float64(batchLen)
		v[k] = beta2t*v[k] + (1-beta2t)*(g*g+epsilon)
		update[k] = g / math.Sqrt(v[k])
	}
	return adafactorClip(update, clipThreshold)
}

// adafactorClip scales the update down so its root-mean-square element is at
// most clipThreshold, and leaves it untouched when it already is. Scaling by
// RMS rather than by the Frobenius norm keeps the cap independent of how many
// parameters the layer happens to hold.
func adafactorClip(update []float64, clipThreshold float64) []float64 {
	if len(update) == 0 || clipThreshold <= 0 {
		return update
	}

	sumSquares := 0.0
	for _, u := range update {
		sumSquares += u * u
	}

	rms := math.Sqrt(sumSquares / float64(len(update)))
	denominator := math.Max(1, rms/clipThreshold)
	if denominator == 1 {
		return update
	}

	for k := range update {
		update[k] /= denominator
	}
	return update
}
