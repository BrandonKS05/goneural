package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// A trained classifier will happily return 0.98 for an input unlike
// anything it was trained on. The functions here address the two separate
// problems that hides.
//
// The first is that a single forward pass reports no spread at all. Monte
// Carlo dropout recovers one by leaving dropout switched on at prediction
// time and sampling repeatedly: each pass is effectively a slightly
// different network, and how much they disagree is a usable measure of how
// much the model's answer depends on which weights happened to survive.
//
// The second is calibration -- whether a stated 90% actually comes true
// nine times in ten. Networks trained to convergence on cross-entropy are
// reliably overconfident, since the loss keeps rewarding sharper outputs
// long after the ranking has stopped improving. ExpectedCalibrationError
// measures the gap and Calibrate fits the one-parameter correction for it.

// PredictSamples runs the forward pass repeatedly with dropout left
// active, returning the per-output mean and standard deviation across
// samples. The mean is the prediction; the standard deviation is how much
// the network disagrees with itself.
//
// It requires HiddenDropout to be set -- with no dropout every pass is
// identical and the spread would be a row of zeros, which is worse than
// useless because it looks like confidence. samples must be at least two.
func (n *NeuralNetwork) PredictSamples(inputs []float64, samples int) (mean, deviation []float64) {
	if n.HiddenDropout <= 0 {
		panic("goneural: PredictSamples needs HiddenDropout set on the network")
	}
	if samples < 2 {
		panic("goneural: PredictSamples needs at least two samples")
	}

	outputs := n.Layers[len(n.Layers)-1].Nodes
	mean = make([]float64, outputs)
	sumSquares := make([]float64, outputs)

	// predict only drops units while the network is marked as training;
	// that flag is exactly what makes this sampling possible, and it is
	// restored afterwards so ordinary Predict stays deterministic.
	n.training = true
	defer func() { n.training = false }()

	for s := 0; s < samples; s++ {
		prediction := n.predict(matrix.NewFromArray(inputs)).Flatten()
		for i, v := range prediction {
			mean[i] += v
			sumSquares[i] += v * v
		}
	}

	deviation = make([]float64, outputs)
	for i := range mean {
		mean[i] /= float64(samples)

		// Rounding can leave a hair of negative variance when every sample
		// agreed exactly.
		variance := math.Max(sumSquares[i]/float64(samples)-mean[i]*mean[i], 0)
		deviation[i] = math.Sqrt(variance)
	}

	return mean, deviation
}

// Entropy is the Shannon entropy of a predicted distribution, in nats: 0
// when the network commits fully to one class, and log(k) when it spreads
// evenly over k of them. Unlike the top probability alone, it accounts for
// the whole distribution -- (0.5, 0.5, 0, 0) and (0.5, 0.2, 0.2, 0.1) are
// equally unsure at the top but not equally uncertain overall.
func Entropy(probabilities []float64) float64 {
	total := 0.0
	for _, p := range probabilities {
		if p > 0 {
			total -= p * math.Log(p)
		}
	}
	return total
}

// PredictiveEntropy is Entropy applied to the network's own output for one
// input, which is the quickest confidence reading available -- one forward
// pass, no sampling. It only means anything for a network whose outputs
// form a distribution, so pair it with a Softmax output layer.
func (n *NeuralNetwork) PredictiveEntropy(inputs []float64) float64 {
	return Entropy(n.Predict(inputs))
}

// ExpectedCalibrationError measures the gap between confidence and
// correctness. Predictions are sorted into `bins` buckets by how confident
// they were; within each bucket the network's mean confidence is compared
// against how often it was actually right, and the absolute gaps are
// averaged, weighted by bucket population.
//
// A result of 0 means a stated 70% comes true 70% of the time. A large one
// usually means overconfidence, though it does not say which way -- read
// Reliability for the direction. bins must be positive; an empty data set
// scores 0.
func (n *NeuralNetwork) ExpectedCalibrationError(dataSet DataSet, bins int) float64 {
	if bins < 1 {
		panic("goneural: ExpectedCalibrationError needs a positive bin count")
	}
	if len(dataSet) == 0 {
		return 0
	}

	confidenceSum := make([]float64, bins)
	correctSum := make([]float64, bins)
	counts := make([]int, bins)

	for _, ds := range dataSet {
		prediction := n.Predict(ds.Inputs)
		confidence, predicted := confidenceOf(prediction)

		// Confidence in [0, 1] indexes the bins; the top edge belongs to
		// the last bucket rather than falling off the end.
		bin := int(confidence * float64(bins))
		if bin >= bins {
			bin = bins - 1
		}
		if bin < 0 {
			bin = 0
		}

		confidenceSum[bin] += confidence
		counts[bin]++
		if predicted == classOf(ds.Targets) {
			correctSum[bin]++
		}
	}

	total := 0.0
	for bin := 0; bin < bins; bin++ {
		if counts[bin] == 0 {
			continue
		}

		meanConfidence := confidenceSum[bin] / float64(counts[bin])
		accuracy := correctSum[bin] / float64(counts[bin])
		total += float64(counts[bin]) * math.Abs(meanConfidence-accuracy)
	}

	return total / float64(len(dataSet))
}

// Reliability returns the network's mean confidence alongside its actual
// accuracy over the data set. Confidence above accuracy is overconfidence,
// the usual direction; below it is the rarer underconfidence. The
// difference is the signed version of what ExpectedCalibrationError
// reports as a magnitude.
func (n *NeuralNetwork) Reliability(dataSet DataSet) (confidence, accuracy float64) {
	if len(dataSet) == 0 {
		return 0, 0
	}

	correct := 0
	for _, ds := range dataSet {
		prediction := n.Predict(ds.Inputs)
		c, predicted := confidenceOf(prediction)

		confidence += c
		if predicted == classOf(ds.Targets) {
			correct++
		}
	}

	return confidence / float64(len(dataSet)), float64(correct) / float64(len(dataSet))
}

// confidenceOf reads a prediction's confidence in its own answer and which
// class that answer is, handling single-output binary networks (where
// confidence is the distance from the 0.5 threshold, rescaled) the same
// way classOf does.
func confidenceOf(prediction []float64) (confidence float64, class int) {
	if len(prediction) == 1 {
		if prediction[0] >= 0.5 {
			return prediction[0], 1
		}
		return 1 - prediction[0], 0
	}

	class = ArgMax(prediction)
	return prediction[class], class
}

// Calibrate fits a temperature that rescales the network's confidence to
// match its accuracy, and returns it. Hold out a data set for this that
// the network was not trained on: fitting on training data would learn
// that the network is right almost always and calibrate toward even more
// confidence.
//
// Temperature scaling (Guo et al., 2017) is the smallest possible fix --
// one parameter, applied to every class alike -- which is exactly why it
// works: it cannot change which class wins, so accuracy is untouched and
// only the confidence moves. A temperature above 1 softens an
// overconfident model; below 1 sharpens an underconfident one.
//
// The search minimizes negative log likelihood over a coarse-to-fine
// sweep, which is ample for a single well-behaved parameter. It requires a
// multi-output network whose predictions form a distribution.
func (n *NeuralNetwork) Calibrate(dataSet DataSet) float64 {
	if n.Layers[len(n.Layers)-1].Nodes < 2 {
		panic("goneural: Calibrate needs a multi-output network")
	}
	if len(dataSet) == 0 {
		panic("goneural: Calibrate needs a non-empty data set")
	}

	// The network's outputs are already a distribution, so the logits it
	// came from are recoverable up to a constant as their logarithm -- and
	// softmax ignores constant shifts, so that is enough to rescale by.
	logits := make([][]float64, len(dataSet))
	truth := make([]int, len(dataSet))
	for i, ds := range dataSet {
		prediction := n.Predict(ds.Inputs)

		logits[i] = make([]float64, len(prediction))
		for j, p := range prediction {
			logits[i][j] = math.Log(math.Max(p, minLogProb))
		}
		truth[i] = classOf(ds.Targets)
	}

	cost := func(temperature float64) float64 {
		total := 0.0
		for i, row := range logits {
			scaled := SoftmaxWithTemperature(temperature, row)
			total -= math.Log(math.Max(scaled[truth[i]], minLogProb))
		}
		return total
	}

	best, bestCost := 1.0, cost(1)
	low, high := 0.05, 10.0

	// Three passes, each narrowing around the previous winner: enough
	// resolution for a parameter whose effect is this smooth.
	for pass := 0; pass < 3; pass++ {
		const points = 40
		step := (high - low) / points

		for i := 0; i <= points; i++ {
			temperature := low + float64(i)*step
			if temperature <= 0 {
				continue
			}
			if c := cost(temperature); c < bestCost {
				best, bestCost = temperature, c
			}
		}

		low = math.Max(0.05, best-step)
		high = best + step
	}

	return best
}

// CalibratedPredict applies a temperature from Calibrate to the network's
// output, returning the corrected distribution. The ranking is unchanged,
// so the predicted class is identical to Predict's -- only the confidence
// moves.
func (n *NeuralNetwork) CalibratedPredict(inputs []float64, temperature float64) []float64 {
	prediction := n.Predict(inputs)

	logits := make([]float64, len(prediction))
	for i, p := range prediction {
		logits[i] = math.Log(math.Max(p, minLogProb))
	}

	return SoftmaxWithTemperature(temperature, logits)
}
