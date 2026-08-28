package goneural

import (
	"math"
	"math/rand"
	"testing"
)

// TestPredictSamplesReportsSpread checks the sampling really varies, that
// the mean is a sensible estimate, and that ordinary Predict is left
// deterministic afterwards.
func TestPredictSamplesReportsSpread(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 16, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)
	n.HiddenDropout = 0.3

	inputs := []float64{0.5, -0.2}
	mean, deviation := n.PredictSamples(inputs, 200)

	if deviation[0] <= 0 {
		t.Errorf("standard deviation %g, want a positive spread under dropout", deviation[0])
	}
	if plain := n.Predict(inputs)[0]; math.Abs(mean[0]-plain) > 0.15 {
		t.Errorf("sampled mean %g is far from the deterministic prediction %g", mean[0], plain)
	}

	// Predict must not have been left in training mode.
	first, second := n.Predict(inputs)[0], n.Predict(inputs)[0]
	if first != second {
		t.Errorf("Predict became non-deterministic: %g then %g", first, second)
	}
}

// TestPredictSamplesConvergesWithMoreSamples pins the estimator: more
// passes should not change the mean much, since it is an average of the
// same distribution.
func TestPredictSamplesConvergesWithMoreSamples(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 16, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)
	n.HiddenDropout = 0.2

	inputs := []float64{0.3, 0.4}
	few, _ := n.PredictSamples(inputs, 500)
	many, _ := n.PredictSamples(inputs, 500)

	if math.Abs(few[0]-many[0]) > 0.05 {
		t.Errorf("two 500-sample estimates differ by %g, want them to agree", math.Abs(few[0]-many[0]))
	}
}

// TestMoreDropoutMeansMoreUncertainty checks the spread tracks the amount
// of dropout, which is what makes it interpretable at all.
func TestMoreDropoutMeansMoreUncertainty(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 16, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	inputs := []float64{0.5, -0.2}

	n.HiddenDropout = 0.05
	_, light := n.PredictSamples(inputs, 400)

	n.HiddenDropout = 0.5
	_, heavy := n.PredictSamples(inputs, 400)

	if heavy[0] <= light[0] {
		t.Errorf("heavy dropout gave spread %g, no more than light dropout's %g", heavy[0], light[0])
	}
}

func TestEntropyMeasuresSpread(t *testing.T) {
	if got := Entropy([]float64{1, 0, 0, 0}); got != 0 {
		t.Errorf("a certain prediction has entropy %g, want 0", got)
	}
	if got, want := Entropy([]float64{0.25, 0.25, 0.25, 0.25}), math.Log(4); math.Abs(got-want) > 1e-12 {
		t.Errorf("a uniform prediction has entropy %g, want log 4 = %g", got, want)
	}

	// Equally unsure at the top, but not equally uncertain overall.
	tie := Entropy([]float64{0.5, 0.5, 0, 0})
	spread := Entropy([]float64{0.5, 0.2, 0.2, 0.1})
	if spread <= tie {
		t.Errorf("the more spread-out distribution scored %g, want above the two-way tie's %g", spread, tie)
	}
}

func TestPredictiveEntropyReadsTheOutput(t *testing.T) {
	n := constantNetwork([]float64{0.7, 0.2, 0.1})

	got := n.PredictiveEntropy([]float64{0})
	want := Entropy([]float64{0.7, 0.2, 0.1})
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("PredictiveEntropy = %g, want %g", got, want)
	}
}

// TestCalibrationErrorSpotsOverconfidence builds two situations with the
// same accuracy and very different honesty about it.
func TestCalibrationErrorSpotsOverconfidence(t *testing.T) {
	// Always says class 0 at 90% confidence, and is right 90% of the time.
	honest := constantNetwork([]float64{0.9, 0.1})
	honestData := make(DataSet, 100)
	for i := range honestData {
		class := 0
		if i%10 == 9 {
			class = 1
		}
		honestData[i] = DataSample{Inputs: []float64{0}, Targets: OneHot(2, class)}
	}

	if got := honest.ExpectedCalibrationError(honestData, 10); got > 0.01 {
		t.Errorf("a well-calibrated network scored %g, want near 0", got)
	}

	// Same answers at 99% confidence, right only half the time.
	overconfident := constantNetwork([]float64{0.99, 0.01})
	mixedData := make(DataSet, 100)
	for i := range mixedData {
		mixedData[i] = DataSample{Inputs: []float64{0}, Targets: OneHot(2, i%2)}
	}

	if got := overconfident.ExpectedCalibrationError(mixedData, 10); math.Abs(got-0.49) > 0.01 {
		t.Errorf("an overconfident network scored %g, want about 0.49", got)
	}

	confidence, accuracy := overconfident.Reliability(mixedData)
	if math.Abs(confidence-0.99) > 1e-9 || math.Abs(accuracy-0.5) > 1e-9 {
		t.Errorf("Reliability = (%g, %g), want (0.99, 0.5)", confidence, accuracy)
	}
	if confidence <= accuracy {
		t.Error("Reliability did not report overconfidence")
	}

	if got := honest.ExpectedCalibrationError(DataSet{}, 10); got != 0 {
		t.Errorf("an empty data set scored %g, want 0", got)
	}
}

// TestCalibrateSoftensAnOverconfidentNetwork is the end-to-end check: a
// network trained to convergence overstates its confidence, and fitting a
// temperature on held-out data must bring the two back together without
// touching accuracy.
func TestCalibrateSoftensAnOverconfidentNetwork(t *testing.T) {
	rand.Seed(5)

	data := twoClusterData(400)
	train, validation := data.Split(0.5)

	n := New(0.1, CrossEntropy(),
		Layer{Nodes: 2},
		Layer{Nodes: 16, Activator: Tanh()},
		Layer{Nodes: 2, Activator: Softmax()},
	)

	// Trained hard on the training half, which is what produces the
	// overconfidence this fixes.
	n.Train(Adam(8), train, 400)

	before := n.ExpectedCalibrationError(validation, 10)
	confidence, accuracy := n.Reliability(validation)
	if confidence <= accuracy {
		t.Skipf("this run did not end up overconfident (confidence %g, accuracy %g)", confidence, accuracy)
	}

	temperature := n.Calibrate(validation)
	if temperature <= 1 {
		t.Errorf("temperature %g, want above 1 to soften an overconfident network", temperature)
	}

	// Rescaled predictions must be better calibrated but rank identically.
	calibratedError := 0.0
	correctBefore, correctAfter := 0, 0
	for _, ds := range validation {
		plain := n.Predict(ds.Inputs)
		scaled := n.CalibratedPredict(ds.Inputs, temperature)

		if ArgMax(plain) != ArgMax(scaled) {
			t.Fatalf("calibration changed the predicted class for %v", ds.Inputs)
		}
		if ArgMax(plain) == classOf(ds.Targets) {
			correctBefore++
		}
		if ArgMax(scaled) == classOf(ds.Targets) {
			correctAfter++
		}
	}
	if correctBefore != correctAfter {
		t.Errorf("accuracy moved from %d to %d correct, want it untouched", correctBefore, correctAfter)
	}

	// Measure the calibrated error the same way ExpectedCalibrationError
	// does, but on the rescaled outputs.
	calibratedError = calibratedECE(n, validation, temperature, 10)
	if calibratedError >= before {
		t.Errorf("calibrated error %g, want below the original %g", calibratedError, before)
	}
}

// calibratedECE mirrors ExpectedCalibrationError over temperature-scaled
// predictions.
func calibratedECE(n *NeuralNetwork, dataSet DataSet, temperature float64, bins int) float64 {
	confidenceSum := make([]float64, bins)
	correctSum := make([]float64, bins)
	counts := make([]int, bins)

	for _, ds := range dataSet {
		scaled := n.CalibratedPredict(ds.Inputs, temperature)
		confidence, predicted := confidenceOf(scaled)

		bin := int(confidence * float64(bins))
		if bin >= bins {
			bin = bins - 1
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
		total += float64(counts[bin]) *
			math.Abs(confidenceSum[bin]/float64(counts[bin])-correctSum[bin]/float64(counts[bin]))
	}

	return total / float64(len(dataSet))
}

// TestCalibrateLeavesACalibratedNetworkAlone checks the fit does not
// invent a correction where none is needed.
func TestCalibrateLeavesACalibratedNetworkAlone(t *testing.T) {
	honest := constantNetwork([]float64{0.9, 0.1})

	data := make(DataSet, 200)
	for i := range data {
		class := 0
		if i%10 == 9 {
			class = 1
		}
		data[i] = DataSample{Inputs: []float64{0}, Targets: OneHot(2, class)}
	}

	if temperature := honest.Calibrate(data); math.Abs(temperature-1) > 0.25 {
		t.Errorf("temperature %g on an already-calibrated network, want near 1", temperature)
	}
}

func TestUncertaintyRejectsBadInput(t *testing.T) {
	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	mustPanicGoneural(t, "no dropout", func() { n.PredictSamples([]float64{0, 0}, 10) })

	n.HiddenDropout = 0.2
	mustPanicGoneural(t, "too few samples", func() { n.PredictSamples([]float64{0, 0}, 1) })
	mustPanicGoneural(t, "zero bins", func() { n.ExpectedCalibrationError(xorData(), 0) })
	mustPanicGoneural(t, "single output", func() { n.Calibrate(xorData()) })

	multi := constantNetwork([]float64{0.5, 0.5})
	mustPanicGoneural(t, "empty data set", func() { multi.Calibrate(DataSet{}) })
}
