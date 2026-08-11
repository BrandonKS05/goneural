package goneural

// ArgMax returns the index of the largest value, taking the first one on
// ties. It panics on an empty slice, which has no largest element.
func ArgMax(values []float64) int {
	if len(values) == 0 {
		panic("goneural: ArgMax of empty slice")
	}

	best := 0
	for i, v := range values[1:] {
		if v > values[best] {
			best = i + 1
		}
	}
	return best
}

// OneHot returns a slice of length size that is all zeros except for a 1 at
// index -- the standard target encoding for classification, and the inverse
// of ArgMax. It panics if index is out of range.
func OneHot(size, index int) []float64 {
	if index < 0 || index >= size {
		panic("goneural: OneHot index out of range")
	}

	out := make([]float64, size)
	out[index] = 1
	return out
}

// Accuracy runs the network over the data set and returns the fraction of
// samples classified correctly, in [0, 1]. Multi-output networks compare
// the ArgMax of the prediction against the ArgMax of the target (one-hot
// labels); single-output networks are treated as binary classifiers with
// the prediction thresholded at 0.5, since the ArgMax of one value says
// nothing. An empty data set has an accuracy of 0.
func (n *NeuralNetwork) Accuracy(dataSet DataSet) float64 {
	if len(dataSet) == 0 {
		return 0
	}

	correct := 0
	for _, ds := range dataSet {
		outputs := n.Predict(ds.Inputs)

		if len(outputs) == 1 {
			if (outputs[0] >= 0.5) == (ds.Targets[0] >= 0.5) {
				correct++
			}
			continue
		}

		if ArgMax(outputs) == ArgMax(ds.Targets) {
			correct++
		}
	}

	return float64(correct) / float64(len(dataSet))
}
