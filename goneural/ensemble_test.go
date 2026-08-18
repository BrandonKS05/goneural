package goneural

import (
	"math"
	"math/rand"
	"testing"
)

func TestEnsemblePredictAveragesMembers(t *testing.T) {
	e := Ensemble{
		constantNetwork([]float64{0.1, 0.9}),
		constantNetwork([]float64{0.3, 0.7}),
		constantNetwork([]float64{0.8, 0.2}),
	}

	got := e.Predict([]float64{0})
	for i, want := range []float64{0.4, 0.6} {
		if math.Abs(got[i]-want) > 1e-12 {
			t.Errorf("Predict()[%d] = %g, want the mean %g", i, got[i], want)
		}
	}

	mustPanicGoneural(t, "empty ensemble", func() { Ensemble{}.Predict([]float64{0}) })
	mustPanicGoneural(t, "mismatched widths", func() {
		Ensemble{
			constantNetwork([]float64{0.5}),
			constantNetwork([]float64{0.5, 0.5}),
		}.Predict([]float64{0})
	})
}

// TestEnsembleVoteTakesTheMajority checks hard voting, and pins the case
// that distinguishes it from averaging: two members that are confidently
// right outvote one that is wildly, loudly wrong.
func TestEnsembleVoteTakesTheMajority(t *testing.T) {
	e := Ensemble{
		constantNetwork([]float64{0.6, 0.4}),
		constantNetwork([]float64{0.55, 0.45}),
		constantNetwork([]float64{0, 100}),
	}

	if got := e.Vote([]float64{0}); got != 0 {
		t.Errorf("Vote() = %d, want the majority's class 0", got)
	}
	// The averaged prediction, in contrast, is dragged over by the outlier.
	if got := classOf(e.Predict([]float64{0})); got != 1 {
		t.Errorf("soft vote = %d, want class 1, which is what makes the hard vote interesting", got)
	}

	mustPanicGoneural(t, "empty ensemble", func() { Ensemble{}.Vote([]float64{0}) })
}

// TestEnsembleVoteBreaksTiesLow pins the documented tie rule, which needs
// to be deterministic despite the count living in a map.
func TestEnsembleVoteBreaksTiesLow(t *testing.T) {
	e := Ensemble{
		constantNetwork([]float64{0.1, 0.9}),
		constantNetwork([]float64{0.9, 0.1}),
	}

	for i := 0; i < 20; i++ {
		if got := e.Vote([]float64{0}); got != 0 {
			t.Fatalf("Vote() = %d on a tie, want the lowest class 0", got)
		}
	}
}

func TestEnsembleAccuracy(t *testing.T) {
	e := Ensemble{
		constantNetwork([]float64{0.1, 0.9}),
		constantNetwork([]float64{0.2, 0.8}),
	}

	data := DataSet{
		{Inputs: []float64{0}, Targets: OneHot(2, 1)},
		{Inputs: []float64{0}, Targets: OneHot(2, 1)},
		{Inputs: []float64{0}, Targets: OneHot(2, 0)},
	}

	if got, want := e.Accuracy(data), 2.0/3.0; math.Abs(got-want) > 1e-12 {
		t.Errorf("Accuracy = %g, want %g", got, want)
	}
	if got := e.Accuracy(DataSet{}); got != 0 {
		t.Errorf("Accuracy of an empty set = %g, want 0", got)
	}
}

// TestBootstrapResamplesWithReplacement checks the size is preserved, every
// drawn sample came from the source, and -- over many draws -- that
// replacement is really happening rather than a shuffle.
func TestBootstrapResamplesWithReplacement(t *testing.T) {
	rand.Seed(7)

	data := DataSet{
		{Inputs: []float64{1}, Targets: []float64{1}},
		{Inputs: []float64{2}, Targets: []float64{0}},
		{Inputs: []float64{3}, Targets: []float64{1}},
		{Inputs: []float64{4}, Targets: []float64{0}},
	}

	sawDuplicate := false
	for trial := 0; trial < 50; trial++ {
		resample := data.Bootstrap()
		if len(resample) != len(data) {
			t.Fatalf("Bootstrap returned %d samples, want %d", len(resample), len(data))
		}

		seen := map[float64]int{}
		for _, ds := range resample {
			if ds.Inputs[0] < 1 || ds.Inputs[0] > 4 {
				t.Fatalf("Bootstrap invented a sample: %v", ds.Inputs)
			}
			seen[ds.Inputs[0]]++
			if seen[ds.Inputs[0]] > 1 {
				sawDuplicate = true
			}
		}
	}

	if !sawDuplicate {
		t.Error("50 bootstrap draws produced no duplicate, so sampling is not with replacement")
	}

	// The source must survive intact for the next member to resample.
	if len(data) != 4 || data[0].Inputs[0] != 1 {
		t.Error("Bootstrap mutated the source data set")
	}
}

// TestBagTrainsIndependentMembers is the end-to-end check: bagging a small
// XOR committee must produce members that differ, leave the prototype
// untrained, and classify the problem correctly as a group.
func TestBagTrainsIndependentMembers(t *testing.T) {
	rand.Seed(7)

	prototype := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)
	untouched := prototype.Weights[0].Copy()

	data := xorData()
	ensemble := Bag(prototype, data, 5, func(n *NeuralNetwork, d DataSet) {
		n.Train(Adam(2), d, 400)
	})

	if len(ensemble) != 5 {
		t.Fatalf("Bag returned %d members, want 5", len(ensemble))
	}

	for i := 0; i < untouched.Rows; i++ {
		for j := 0; j < untouched.Columns; j++ {
			if got, want := prototype.Weights[0].At(i, j), untouched.At(i, j); got != want {
				t.Fatalf("prototype weight (%d, %d) = %g, want the untouched %g", i, j, got, want)
			}
		}
	}

	if ensemble[0].Weights[0].At(0, 0) == ensemble[1].Weights[0].At(0, 0) {
		t.Error("two members share a weight exactly, so they were not independently initialized")
	}

	// Bagging drops roughly a third of the samples from each member's view,
	// which on a four-sample problem can cost an individual member a case;
	// the committee is what has to be right.
	if got := ensemble.Accuracy(data); got != 1 {
		t.Errorf("ensemble accuracy = %g, want 1", got)
	}
}

func TestBagRejectsBadParameters(t *testing.T) {
	prototype := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)
	train := func(n *NeuralNetwork, d DataSet) {}

	mustPanicGoneural(t, "zero members", func() { Bag(prototype, xorData(), 0, train) })
	mustPanicGoneural(t, "empty data set", func() { Bag(prototype, DataSet{}, 3, train) })
	mustPanicGoneural(t, "missing train function", func() { Bag(prototype, xorData(), 3, nil) })
}
