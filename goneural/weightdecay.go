package goneural

import "fmt"

// WithWeightDecay wraps any optimizer with decoupled weight decay: after
// each invocation (one epoch), every weight shrinks toward zero by
// decayRate * w. Pulling the whole weight matrix in like this is the
// decoupled form of L2 regularization -- it penalizes large weights
// without routing the penalty through the gradients, so it composes with
// any inner optimizer unchanged. Biases are left alone, as is
// conventional: they don't multiply inputs, so shrinking them doesn't
// simplify the learned function.
//
// Note the granularity difference from AdamW and Lion, whose built-in
// decay applies per batch and scales with the learning rate; this wrapper
// applies decayRate once per epoch regardless of batch count, so
// comparable behaviour needs a correspondingly smaller rate.
func WithWeightDecay(optimizer Optimizer, decayRate float64) Optimizer {
	if decayRate < 0 || decayRate >= 1 {
		panic(fmt.Sprintf("goneural: weight decay rate %g outside [0, 1)", decayRate))
	}

	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		err := optimizer(n, dataSet)

		if decayRate > 0 {
			for l := range n.Weights {
				n.Weights[l] = n.Weights[l].Scale(1 - decayRate)
			}
		}

		return err
	}
}
