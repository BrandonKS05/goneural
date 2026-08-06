package goneural

import "github.com/BrandonKS05/goneural/matrix"

// Optimizer is the optimizer type (returns the error).
type Optimizer func(n *NeuralNetwork, dataSet DataSet) float64

// SGD is Stochastic Gradient Descent (On-line Training)
func SGD() Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		return MBGD(1)(n, dataSet)
	}
}

// MBGD is a Mini-Batch Gradient Descent (Batch training)
func MBGD(batchSize int) Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		err := 0.0

		for i := 0; i < len(dataSet); i += batchSize {
			batch := dataSet.Batch(i, batchSize)
			if len(batch) == 0 {
				continue
			}

			lenLayers := len(n.Layers)
			lenWeights := lenLayers - 1

			deltas := make([]matrix.Matrix, lenWeights)
			gradients := make([]matrix.Matrix, lenWeights)

			// Zero the matrices
			for i := 0; i < lenWeights; i++ {
				deltas[i] = matrix.
					New(n.Weights[i].Rows, n.Weights[i].Columns, nil)
				gradients[i] = matrix.
					New(n.Biases[i].Rows, n.Biases[i].Columns, nil)
			}

			for _, ds := range batch {
				inputs := matrix.NewFromArray(ds.Inputs)
				targets := matrix.NewFromArray(ds.Targets)
				outputs := n.predict(inputs)

				err += n.Loss.F(outputs, targets)

				// delta is the error signal at the current layer, with that
				// layer's own activation derivative already folded in. It
				// starts at the output layer and gets propagated backward
				// (and re-folded with each layer's activation derivative)
				// as the loop descends, per the standard backprop
				// recurrence delta[k] = (W[k]^T . delta[k+1]) * act'[k](a[k]).
				delta := outputDelta(n, outputs, targets)

				for i := lenWeights - 1; i >= 0; i-- {
					currentGradients := delta.Scale(n.LearningRate)
					currentDeltas := delta.
						Multiply(n.Activations[i].Transpose()).
						Scale(n.LearningRate)

					deltas[i] = deltas[i].AddMatrix(currentDeltas)
					gradients[i] = gradients[i].AddMatrix(currentGradients)

					if i > 0 {
						delta = n.Weights[i].Transpose().Multiply(delta)
						delta = n.Activations[i].
							Map(func(val float64, x, y int) float64 {
								return n.Layers[i].Activator.FPrime(val)
							}).
							HadamardProduct(delta)
					}
				}
			}

			for i := 0; i < lenWeights; i++ {
				deltas[i] = deltas[i].Map(
					func(val float64, x, y int) float64 {
						return val / float64(len(batch)) // average the changes
					},
				)

				gradients[i] = gradients[i].Map(
					func(val float64, x, y int) float64 {
						return val / float64(len(batch)) // average the changes
					},
				)

				n.Weights[i] = n.Weights[i].AddMatrix(deltas[i])
				n.Biases[i] = n.Biases[i].AddMatrix(gradients[i])
			}
		}

		return err
	}
}

// outputDelta computes the initial backward error signal at the output
// layer, in the "add this direction to descend" convention MBGD, Adam, and
// ConcurrentMBGD all share. Softmax paired with CrossEntropy gets the
// well-known combined shortcut (target - prediction) directly -- deriving
// the full softmax Jacobian generically isn't worth it for this library --
// everything else goes through the generic Loss.FPrime + activation
// derivative chain rule.
func outputDelta(n *NeuralNetwork, outputs, targets matrix.Matrix) matrix.Matrix {
	outputActivator := n.Layers[len(n.Layers)-1].Activator

	if outputActivator.Name == softmax {
		if n.Loss.Name != crossEntropy {
			panic("goneural: softmax output activation requires CrossEntropy loss")
		}
		return targets.SubtractMatrix(outputs)
	}

	return outputs.
		Map(func(val float64, x, y int) float64 {
			return outputActivator.FPrime(val)
		}).
		HadamardProduct(n.Loss.FPrime(outputs, targets))
}

// GD is a normal gradient descent (Optimizes after the entire data set)
func GD() Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		return MBGD(len(dataSet))(n, dataSet)
	}
}
