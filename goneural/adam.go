package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// Adam is the Adam (Adaptive Moment Estimation) optimizer: for every weight
// and bias it keeps a running estimate of the gradient's mean (first
// moment) and uncentered variance (second moment), and uses them to give
// each parameter its own step size. This usually trains faster and more
// stably than plain gradient descent, especially on parameters that get
// noisy or infrequent gradients. Uses the standard defaults from the
// original paper (Kingma & Ba, 2014): beta1=0.9, beta2=0.999, epsilon=1e-8.
func Adam(batchSize int) Optimizer {
	const (
		beta1   = 0.9
		beta2   = 0.999
		epsilon = 1e-8
	)

	// First (m) and second (v) moment estimates, one flattened slice per
	// weight/bias layer. Captured in the closure so they persist across
	// the many epochs Train() calls this Optimizer for; lazily sized on
	// the first call since that's the first point we have a *NeuralNetwork.
	var mW, vW, mB, vB [][]float64
	t := 0

	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		lenWeights := len(n.Weights)

		if mW == nil {
			mW = make([][]float64, lenWeights)
			vW = make([][]float64, lenWeights)
			mB = make([][]float64, lenWeights)
			vB = make([][]float64, lenWeights)
			for l := 0; l < lenWeights; l++ {
				mW[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
				vW[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
				mB[l] = make([]float64, n.Biases[l].Rows)
				vB[l] = make([]float64, n.Biases[l].Rows)
			}
		}

		err := 0.0

		for i := 0; i < len(dataSet); i += batchSize {
			batch := dataSet.Batch(i, batchSize)
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

				// Same backprop recurrence as MBGD, but Adam needs the true
				// (positive) gradient dLoss/dw rather than MBGD's
				// pre-negated descent direction, since it works out its own
				// per-parameter step size from the raw gradient.
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

			t++
			biasCorrection1 := 1 - math.Pow(beta1, float64(t))
			biasCorrection2 := 1 - math.Pow(beta2, float64(t))

			for l := 0; l < lenWeights; l++ {
				rows := n.Layers[l+1].Nodes
				cols := n.Layers[l].Nodes

				wStep := adamStep(weightGrads[l].Flatten(), mW[l], vW[l], len(batch), n.LearningRate, beta1, beta2, epsilon, biasCorrection1, biasCorrection2)
				bStep := adamStep(biasGrads[l].Flatten(), mB[l], vB[l], len(batch), n.LearningRate, beta1, beta2, epsilon, biasCorrection1, biasCorrection2)

				n.Weights[l] = n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, wStep))
				n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, bStep))
			}
		}

		return err
	}
}

// adamStep updates the moment estimates m/v in place from a batch's summed
// gradient and returns the per-parameter step to subtract from the weights.
func adamStep(gradSum, m, v []float64, batchLen int, lr, beta1, beta2, epsilon, biasCorrection1, biasCorrection2 float64) []float64 {
	step := make([]float64, len(gradSum))
	for k, g := range gradSum {
		g /= float64(batchLen)
		m[k] = beta1*m[k] + (1-beta1)*g
		v[k] = beta2*v[k] + (1-beta2)*g*g
		mHat := m[k] / biasCorrection1
		vHat := v[k] / biasCorrection2
		step[k] = lr * mHat / (math.Sqrt(vHat) + epsilon)
	}
	return step
}
