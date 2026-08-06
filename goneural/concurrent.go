package goneural

import (
	"sync"

	"github.com/BrandonKS05/goneural/matrix"
)

// ConcurrentMBGD is Mini-Batch Gradient Descent that computes each sample's
// gradient in its own goroutine, then reduces them once the whole batch has
// finished. The sequential optimizers reuse NeuralNetwork.Activations as
// scratch space during their forward pass, which multiple goroutines
// writing to at once would race on -- so each goroutine here keeps its own
// local activations instead of touching that shared slice.
func ConcurrentMBGD(batchSize int) Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		err := 0.0
		lenWeights := len(n.Weights)

		for i := 0; i < len(dataSet); i += batchSize {
			batch := dataSet.Batch(i, batchSize)
			if len(batch) == 0 {
				continue
			}

			type sampleResult struct {
				err         float64
				weightGrads []matrix.Matrix
				biasGrads   []matrix.Matrix
			}

			results := make([]sampleResult, len(batch))
			var wg sync.WaitGroup

			for s, ds := range batch {
				wg.Add(1)
				go func(s int, ds DataSample) {
					defer wg.Done()

					activations := forwardPass(n, matrix.NewFromArray(ds.Inputs))
					outputs := activations[lenWeights]
					targets := matrix.NewFromArray(ds.Targets)

					weightGrads, biasGrads := backwardGradients(n, activations, targets)

					results[s] = sampleResult{
						err:         n.Loss.F(outputs, targets),
						weightGrads: weightGrads,
						biasGrads:   biasGrads,
					}
				}(s, ds)
			}

			wg.Wait()

			deltas := make([]matrix.Matrix, lenWeights)
			gradients := make([]matrix.Matrix, lenWeights)
			for l := 0; l < lenWeights; l++ {
				deltas[l] = matrix.New(n.Weights[l].Rows, n.Weights[l].Columns, nil)
				gradients[l] = matrix.New(n.Biases[l].Rows, n.Biases[l].Columns, nil)
			}

			for _, res := range results {
				err += res.err
				for l := 0; l < lenWeights; l++ {
					deltas[l] = deltas[l].AddMatrix(res.weightGrads[l])
					gradients[l] = gradients[l].AddMatrix(res.biasGrads[l])
				}
			}

			for l := 0; l < lenWeights; l++ {
				deltas[l] = deltas[l].Map(func(val float64, x, y int) float64 {
					return val * n.LearningRate / float64(len(batch))
				})
				gradients[l] = gradients[l].Map(func(val float64, x, y int) float64 {
					return val * n.LearningRate / float64(len(batch))
				})

				n.Weights[l] = n.Weights[l].AddMatrix(deltas[l])
				n.Biases[l] = n.Biases[l].AddMatrix(gradients[l])
			}
		}

		return err
	}
}

// forwardPass mirrors NeuralNetwork.predict but returns a fresh activations
// slice instead of writing into n.Activations, so it's safe to call from
// multiple goroutines at once.
func forwardPass(n *NeuralNetwork, input matrix.Matrix) []matrix.Matrix {
	activations := make([]matrix.Matrix, len(n.Layers))
	activations[0] = input

	mat := input
	for i := 0; i < len(n.Weights); i++ {
		mat = n.Weights[i].Multiply(mat).AddMatrix(n.Biases[i]).Map(func(val float64, x, y int) float64 {
			return n.Layers[i+1].Activator.F(val)
		})
		activations[i+1] = mat
	}

	return activations
}

// backwardGradients mirrors MBGD's per-sample backprop (same delta[k] =
// (W[k]^T . delta[k+1]) * act'[k](a[k]) recurrence, same sign convention --
// the result is the "add this to descend" direction, unscaled by learning
// rate, ready to be summed and averaged like MBGD's deltas/gradients), but
// reads from a local activations slice instead of n.Activations, so it's
// safe to call concurrently.
func backwardGradients(n *NeuralNetwork, activations []matrix.Matrix, targets matrix.Matrix) ([]matrix.Matrix, []matrix.Matrix) {
	lenWeights := len(n.Weights)
	weightGrads := make([]matrix.Matrix, lenWeights)
	biasGrads := make([]matrix.Matrix, lenWeights)

	outputs := activations[lenWeights]
	delta := outputDelta(n, outputs, targets)

	for i := lenWeights - 1; i >= 0; i-- {
		weightGrads[i] = delta.Multiply(activations[i].Transpose())
		biasGrads[i] = delta

		if i > 0 {
			delta = n.Weights[i].Transpose().Multiply(delta)
			delta = activations[i].
				Map(func(val float64, x, y int) float64 {
					return n.Layers[i].Activator.FPrime(val)
				}).
				HadamardProduct(delta)
		}
	}

	return weightGrads, biasGrads
}
