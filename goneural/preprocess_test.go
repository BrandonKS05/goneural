package goneural

import (
	"math"
	"math/rand"
	"testing"
)

func scalerData() DataSet {
	return DataSet{
		{Inputs: []float64{1, 100, 7}, Targets: []float64{0}},
		{Inputs: []float64{2, 300, 7}, Targets: []float64{1}},
		{Inputs: []float64{3, 200, 7}, Targets: []float64{1}},
		{Inputs: []float64{6, 400, 7}, Targets: []float64{0}},
	}
}

// TestStandardizerCentersAndScales checks the fitted transform actually
// produces zero mean and unit variance per feature.
func TestStandardizerCentersAndScales(t *testing.T) {
	data := scalerData()
	scaled := FitStandardizer(data).Transform(data)

	for feature := 0; feature < 2; feature++ { // the third is constant
		mean, sqSum := 0.0, 0.0
		for _, ds := range scaled {
			mean += ds.Inputs[feature]
		}
		mean /= float64(len(scaled))

		for _, ds := range scaled {
			sqSum += (ds.Inputs[feature] - mean) * (ds.Inputs[feature] - mean)
		}
		std := math.Sqrt(sqSum / float64(len(scaled)))

		if math.Abs(mean) > 1e-12 {
			t.Errorf("feature %d: mean = %g, want 0", feature, mean)
		}
		if math.Abs(std-1) > 1e-12 {
			t.Errorf("feature %d: std = %g, want 1", feature, std)
		}
	}
}

// TestMinMaxScalerMapsOntoUnitRange pins the endpoints of the fitted range.
func TestMinMaxScalerMapsOntoUnitRange(t *testing.T) {
	data := scalerData()
	scaled := FitMinMaxScaler(data).Transform(data)

	for feature := 0; feature < 2; feature++ {
		min, max := math.Inf(1), math.Inf(-1)
		for _, ds := range scaled {
			min = math.Min(min, ds.Inputs[feature])
			max = math.Max(max, ds.Inputs[feature])
		}
		if math.Abs(min) > 1e-12 || math.Abs(max-1) > 1e-12 {
			t.Errorf("feature %d: range [%g, %g], want [0, 1]", feature, min, max)
		}
	}
}

// TestScalerHandlesConstantFeature guards the divide-by-zero: a feature
// that never varies must map to a finite constant, not NaN.
func TestScalerHandlesConstantFeature(t *testing.T) {
	data := scalerData()

	for name, s := range map[string]*Scaler{
		"standardizer": FitStandardizer(data),
		"min-max":      FitMinMaxScaler(data),
	} {
		for _, ds := range s.Transform(data) {
			if got := ds.Inputs[2]; math.IsNaN(got) || got != 0 {
				t.Errorf("%s: constant feature = %g, want 0", name, got)
			}
		}
	}
}

// TestScalerLeavesSourceDataAlone makes sure Transform copies rather than
// rewriting the caller's samples in place.
func TestScalerLeavesSourceDataAlone(t *testing.T) {
	data := scalerData()
	FitStandardizer(data).Transform(data)

	if got := data[0].Inputs[1]; got != 100 {
		t.Fatalf("source input mutated to %g, want 100", got)
	}
}

// TestScalerRoundTrips checks InverseInputs undoes TransformInputs.
func TestScalerRoundTrips(t *testing.T) {
	data := scalerData()

	for name, s := range map[string]*Scaler{
		"standardizer": FitStandardizer(data),
		"min-max":      FitMinMaxScaler(data),
	} {
		in := []float64{2.5, 250, 7}
		back := s.InverseInputs(s.TransformInputs(in))
		for i := range in {
			if math.Abs(back[i]-in[i]) > 1e-12 {
				t.Errorf("%s: round trip of feature %d = %g, want %g", name, i, back[i], in[i])
			}
		}
	}
}

func TestScalerRejectsBadInput(t *testing.T) {
	mustPanicGoneural(t, "empty data set", func() { FitStandardizer(DataSet{}) })
	mustPanicGoneural(t, "empty data set", func() { FitMinMaxScaler(DataSet{}) })
	mustPanicGoneural(t, "ragged samples", func() {
		FitStandardizer(DataSet{
			{Inputs: []float64{1}, Targets: []float64{0}},
			{Inputs: []float64{1, 2}, Targets: []float64{0}},
		})
	})
	mustPanicGoneural(t, "wrong width", func() {
		FitStandardizer(scalerData()).TransformInputs([]float64{1})
	})
	mustPanicGoneural(t, "wrong width", func() {
		FitStandardizer(scalerData()).InverseInputs([]float64{1})
	})
}

// TestMixupProducesConvexCombinations checks every blended sample sits on
// the segment between two originals, with inputs and targets sharing one
// mixing weight.
func TestMixupProducesConvexCombinations(t *testing.T) {
	rand.Seed(7)

	data := DataSet{
		{Inputs: []float64{0, 0}, Targets: []float64{1, 0}},
		{Inputs: []float64{1, 10}, Targets: []float64{0, 1}},
	}

	mixed := data.Mixup(1)
	if len(mixed) != len(data) {
		t.Fatalf("Mixup returned %d samples, want %d", len(mixed), len(data))
	}

	for _, ds := range mixed {
		// Every input is (1-lambda) of the way from sample 0 to sample 1,
		// so the two features must agree on that fraction, and the targets
		// must sit at the same point on their own segment.
		lambda := ds.Inputs[0]
		if lambda < 0 || lambda > 1 {
			t.Fatalf("mixing weight %g outside [0, 1]", lambda)
		}
		if got, want := ds.Inputs[1], 10*lambda; math.Abs(got-want) > 1e-12 {
			t.Errorf("second feature = %g, want %g (same lambda as the first)", got, want)
		}
		if got, want := ds.Targets[1], lambda; math.Abs(got-want) > 1e-12 {
			t.Errorf("target = %g, want %g (same lambda as the inputs)", got, want)
		}
		if sum := ds.Targets[0] + ds.Targets[1]; math.Abs(sum-1) > 1e-12 {
			t.Errorf("blended one-hot targets sum to %g, want 1", sum)
		}
	}
}

// TestMixupLeavesSourceDataAlone pins that the originals survive, so the
// same set can be re-mixed every epoch.
func TestMixupLeavesSourceDataAlone(t *testing.T) {
	rand.Seed(7)

	data := scalerData()
	data.Mixup(0.4)

	if got := data[0].Inputs[0]; got != 1 {
		t.Fatalf("source input mutated to %g, want 1", got)
	}
}

// TestBetaSampleMatchesDistribution checks the Beta draws stay in range and
// that alpha shifts their spread the way the distribution says it should:
// Beta(a, a) has mean 1/2 and variance 1/(8a + 4), so a large alpha
// concentrates around the even mix and a small one spreads to the ends.
func TestBetaSampleMatchesDistribution(t *testing.T) {
	rand.Seed(7)

	const draws = 20000

	for _, alpha := range []float64{0.2, 1, 5} {
		mean, sqSum := 0.0, 0.0
		samples := make([]float64, draws)

		for i := range samples {
			v := betaSample(alpha, alpha)
			if v < 0 || v > 1 || math.IsNaN(v) {
				t.Fatalf("alpha %g: draw %g outside [0, 1]", alpha, v)
			}
			samples[i] = v
			mean += v
		}
		mean /= draws

		for _, v := range samples {
			sqSum += (v - mean) * (v - mean)
		}
		variance := sqSum / draws

		if math.Abs(mean-0.5) > 0.02 {
			t.Errorf("alpha %g: mean = %g, want 0.5", alpha, mean)
		}
		if want := 1 / (8*alpha + 4); math.Abs(variance-want) > 0.02 {
			t.Errorf("alpha %g: variance = %g, want %g", alpha, variance, want)
		}
	}
}

func TestMixupRejectsBadInput(t *testing.T) {
	mustPanicGoneural(t, "non-positive alpha", func() { scalerData().Mixup(0) })
	mustPanicGoneural(t, "ragged samples", func() {
		DataSet{
			{Inputs: []float64{1}, Targets: []float64{0}},
			{Inputs: []float64{1, 2}, Targets: []float64{0}},
		}.Mixup(1)
	})

	if got := (DataSet{}).Mixup(1); len(got) != 0 {
		t.Errorf("Mixup of an empty set returned %d samples, want 0", len(got))
	}
}
