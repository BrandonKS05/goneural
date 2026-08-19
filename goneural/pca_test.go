package goneural

import (
	"math"
	"math/rand"
	"testing"
)

// lineData returns points spread along a 45-degree line with a little
// jitter across it, so the leading principal axis is known in advance:
// (1, 1)/sqrt(2), carrying nearly all the variance.
func lineData(n int, jitter float64) DataSet {
	data := make(DataSet, n)
	for i := range data {
		t := float64(i)/float64(n-1)*10 - 5
		off := rand.NormFloat64() * jitter

		data[i] = DataSample{
			Inputs:  []float64{t + off, t - off},
			Targets: []float64{t},
		}
	}
	return data
}

// TestPCAFindsTheDirectionOfVariance checks the leading axis lands on the
// line the data lies along, and that it accounts for essentially all the
// variance.
func TestPCAFindsTheDirectionOfVariance(t *testing.T) {
	rand.Seed(7)

	p := FitPCA(lineData(200, 0.05), 2)

	root := 1 / math.Sqrt(2)
	// A loose tolerance on purpose: the axis is estimated from jittered
	// samples, so it lands near the true 45 degrees, not exactly on it.
	if math.Abs(math.Abs(p.Components.At(0, 0))-root) > 0.01 ||
		math.Abs(math.Abs(p.Components.At(0, 1))-root) > 0.01 {
		t.Errorf("leading axis = (%g, %g), want +-(%g, %g)",
			p.Components.At(0, 0), p.Components.At(0, 1), root, root)
	}

	ratio := p.ExplainedVarianceRatio()
	if ratio[0] < 0.99 {
		t.Errorf("leading component explains %g of the variance, want nearly all of it", ratio[0])
	}
	if sum := ratio[0] + ratio[1]; math.Abs(sum-1) > 1e-9 {
		t.Errorf("the two components explain %g of the variance, want 1", sum)
	}
}

// TestPCADecorrelatesAndOrders is the defining property: in the projected
// space the features are uncorrelated, and their variances match the
// reported ones in descending order.
func TestPCADecorrelatesAndOrders(t *testing.T) {
	rand.Seed(7)

	data := make(DataSet, 300)
	for i := range data {
		// Three correlated features built from two underlying factors.
		a, b := rand.NormFloat64(), rand.NormFloat64()
		data[i] = DataSample{
			Inputs:  []float64{a + 0.1*b, 2*a - b, a + b},
			Targets: []float64{0},
		}
	}

	p := FitPCA(data, 3)
	projected := p.Transform(data)

	mean := make([]float64, 3)
	for _, ds := range projected {
		for i, v := range ds.Inputs {
			mean[i] += v
		}
	}
	for i := range mean {
		mean[i] /= float64(len(projected))
		if math.Abs(mean[i]) > 1e-9 {
			t.Errorf("projected feature %d has mean %g, want 0", i, mean[i])
		}
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			covariance := 0.0
			for _, ds := range projected {
				covariance += (ds.Inputs[i] - mean[i]) * (ds.Inputs[j] - mean[j])
			}
			covariance /= float64(len(projected))

			if i == j {
				if math.Abs(covariance-p.Variances[i]) > 1e-9 {
					t.Errorf("projected feature %d has variance %g, want the reported %g",
						i, covariance, p.Variances[i])
				}
				continue
			}
			if math.Abs(covariance) > 1e-9 {
				t.Errorf("projected features %d and %d have covariance %g, want 0", i, j, covariance)
			}
		}
	}

	for i := 1; i < len(p.Variances); i++ {
		if p.Variances[i] > p.Variances[i-1] {
			t.Errorf("variance %d (%g) exceeds the preceding %g", i, p.Variances[i], p.Variances[i-1])
		}
	}
}

// TestPCARoundTripsWithEveryComponent pins that a full-rank projection
// loses nothing.
func TestPCARoundTripsWithEveryComponent(t *testing.T) {
	rand.Seed(7)

	data := lineData(50, 0.4)
	p := FitPCA(data, 2)

	for _, ds := range data {
		back := p.InverseInputs(p.TransformInputs(ds.Inputs))
		for i := range ds.Inputs {
			if math.Abs(back[i]-ds.Inputs[i]) > 1e-9 {
				t.Fatalf("round trip of %v gave %v", ds.Inputs, back)
			}
		}
	}
}

// TestPCAReductionKeepsTheSignal drops to one component and checks the
// reconstruction still tracks the data closely -- which is only true
// because the discarded axis held almost nothing.
func TestPCAReductionKeepsTheSignal(t *testing.T) {
	rand.Seed(7)

	data := lineData(200, 0.05)
	p := FitPCA(data, 1)

	reduced := p.Transform(data)
	if got := len(reduced[0].Inputs); got != 1 {
		t.Fatalf("reduced sample has %d features, want 1", got)
	}

	worst := 0.0
	for i, ds := range data {
		back := p.InverseInputs(reduced[i].Inputs)
		for f := range ds.Inputs {
			worst = math.Max(worst, math.Abs(back[f]-ds.Inputs[f]))
		}
	}
	if worst > 0.5 {
		t.Errorf("worst reconstruction error after dropping an axis is %g, want it small", worst)
	}
}

// TestPCAWhitensToUnitVariance checks the optional scaling step.
func TestPCAWhitensToUnitVariance(t *testing.T) {
	rand.Seed(7)

	data := lineData(300, 0.3)
	p := FitPCA(data, 2)
	p.Whiten = true

	projected := p.Transform(data)
	for feature := 0; feature < 2; feature++ {
		mean, sqSum := 0.0, 0.0
		for _, ds := range projected {
			mean += ds.Inputs[feature]
		}
		mean /= float64(len(projected))

		for _, ds := range projected {
			sqSum += (ds.Inputs[feature] - mean) * (ds.Inputs[feature] - mean)
		}

		if variance := sqSum / float64(len(projected)); math.Abs(variance-1) > 1e-9 {
			t.Errorf("whitened feature %d has variance %g, want 1", feature, variance)
		}
	}

	// Whitening must stay invertible.
	back := p.InverseInputs(p.TransformInputs(data[0].Inputs))
	for i := range data[0].Inputs {
		if math.Abs(back[i]-data[0].Inputs[i]) > 1e-9 {
			t.Fatalf("whitened round trip of %v gave %v", data[0].Inputs, back)
		}
	}
}

// TestPCAHandlesConstantFeature guards the degenerate axis: a feature that
// never varies contributes a zero eigenvalue, which whitening must not
// divide by.
func TestPCAHandlesConstantFeature(t *testing.T) {
	rand.Seed(7)

	data := make(DataSet, 50)
	for i := range data {
		data[i] = DataSample{
			Inputs:  []float64{float64(i), 7},
			Targets: []float64{0},
		}
	}

	p := FitPCA(data, 2)
	p.Whiten = true

	for _, ds := range p.Transform(data) {
		for i, v := range ds.Inputs {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("projected feature %d = %g on a constant input", i, v)
			}
		}
	}

	if ratio := p.ExplainedVarianceRatio(); math.Abs(ratio[0]-1) > 1e-9 || math.Abs(ratio[1]) > 1e-9 {
		t.Errorf("variance ratios %v, want everything on the first axis", ratio)
	}
}

// TestPCASignsAreStable checks the sign convention: refitting the same
// data must not flip an axis.
func TestPCASignsAreStable(t *testing.T) {
	rand.Seed(7)
	data := lineData(100, 0.2)

	first := FitPCA(data, 2)
	second := FitPCA(data, 2)

	for k := 0; k < 2; k++ {
		for f := 0; f < 2; f++ {
			if first.Components.At(k, f) != second.Components.At(k, f) {
				t.Fatalf("component %d differs between fits: %g vs %g",
					k, first.Components.At(k, f), second.Components.At(k, f))
			}
		}

		// Each axis's largest entry is positive, by construction.
		largest, sign := 0.0, 0.0
		for f := 0; f < 2; f++ {
			if v := first.Components.At(k, f); math.Abs(v) > largest {
				largest, sign = math.Abs(v), v
			}
		}
		if sign < 0 {
			t.Errorf("component %d has a negative leading entry %g", k, sign)
		}
	}
}

// TestPCAFeedsANetwork is the end-to-end use: reduce the inputs, then
// train on the reduced data.
func TestPCAFeedsANetwork(t *testing.T) {
	rand.Seed(7)

	// XOR with two redundant copies of each input, which PCA should be
	// able to collapse back to two meaningful axes.
	data := make(DataSet, 0, 4)
	for _, ds := range xorData() {
		data = append(data, DataSample{
			Inputs:  []float64{ds.Inputs[0], ds.Inputs[1], ds.Inputs[0], ds.Inputs[1]},
			Targets: ds.Targets,
		})
	}

	p := FitPCA(data, 2)
	reduced := p.Transform(data)

	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	optimizer := Adam(2)
	for i := 0; i < 800; i++ {
		reduced.Shuffle()
		optimizer(n, reduced)
	}

	if got := n.Accuracy(reduced); got != 1 {
		t.Errorf("accuracy on the reduced data = %g, want 1", got)
	}
}

func TestPCARejectsBadInput(t *testing.T) {
	data := lineData(10, 0.1)

	mustPanicGoneural(t, "too many components", func() { FitPCA(data, 3) })
	mustPanicGoneural(t, "zero components", func() { FitPCA(data, 0) })
	mustPanicGoneural(t, "empty data set", func() { FitPCA(DataSet{}, 1) })
	mustPanicGoneural(t, "single sample", func() {
		FitPCA(DataSet{{Inputs: []float64{1, 2}, Targets: []float64{0}}}, 1)
	})
	mustPanicGoneural(t, "ragged samples", func() {
		FitPCA(DataSet{
			{Inputs: []float64{1}, Targets: []float64{0}},
			{Inputs: []float64{1, 2}, Targets: []float64{0}},
		}, 1)
	})

	p := FitPCA(data, 1)
	mustPanicGoneural(t, "wrong input width", func() { p.TransformInputs([]float64{1}) })
	mustPanicGoneural(t, "wrong projected width", func() { p.InverseInputs([]float64{1, 2}) })
}
