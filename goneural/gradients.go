package goneural

import "github.com/BrandonKS05/goneural/matrix"

// accumulateBatchGradients backpropagates every sample in the batch and
// returns the layer-by-layer *summed* true gradients dLoss/dW and dLoss/db,
// along with the batch's summed loss. Every stateful optimizer (the Adam
// family, momentum, RMSProp and the rest) shares this exact recurrence and
// differs only in how it turns the averaged gradient into a step, so the
// recurrence lives here once. Dividing by the batch size is left to the
// caller.
func accumulateBatchGradients(n *NeuralNetwork, batch DataSet) (weightGrads, biasGrads []matrix.Matrix, err float64) {
	lenWeights := len(n.Weights)

	weightGrads = make([]matrix.Matrix, lenWeights)
	biasGrads = make([]matrix.Matrix, lenWeights)
	for l := 0; l < lenWeights; l++ {
		weightGrads[l] = matrix.New(n.Weights[l].Rows, n.Weights[l].Columns, nil)
		biasGrads[l] = matrix.New(n.Biases[l].Rows, n.Biases[l].Columns, nil)
	}

	for _, ds := range batch {
		outputs := n.predict(matrix.NewFromArray(ds.Inputs))
		targets := matrix.NewFromArray(ds.Targets)
		err += n.Loss.F(outputs, targets)

		// outputDelta yields MBGD's pre-negated "add to descend" direction;
		// undoing the sign gives the true gradient the adaptive optimizers
		// build their own steps from.
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

	return weightGrads, biasGrads, err
}
