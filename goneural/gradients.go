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

	// Mark the forward passes below as training so predict may apply
	// dropout; Predict calls from user code stay inference-only.
	n.training = true
	defer func() { n.training = false }()

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
				delta = activationFold(n, l).HadamardProduct(delta)
			}
		}
	}

	return weightGrads, biasGrads, err
}

// activationFold returns the elementwise factor that carries delta through
// layer l's activation. Without dropout that is FPrime evaluated on the
// stored activation, per the package convention. When this sample dropped
// units in layer l, the stored activation is the mask-scaled value
// y~ = mask * a / keep, so the true derivative chain is recovered by
// undoing the scale before FPrime (a = y~ * keep for survivors), masking
// out the dropped units entirely, and re-applying the 1/keep from the
// forward scaling.
func activationFold(n *NeuralNetwork, l int) matrix.Matrix {
	if n.HiddenDropout > 0 && n.dropMasks != nil && n.dropMasks[l].Rows > 0 {
		keep := 1 - n.HiddenDropout
		return n.Activations[l].
			Map(func(val float64, x, y int) float64 {
				return n.Layers[l].Activator.FPrime(val * keep)
			}).
			HadamardProduct(n.dropMasks[l]).
			Scale(1 / keep)
	}

	return n.Activations[l].Map(func(val float64, x, y int) float64 {
		return n.Layers[l].Activator.FPrime(val)
	})
}
