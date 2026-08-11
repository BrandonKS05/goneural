package goneural

import (
	"math"
	"math/rand"
	"testing"
)

// TestPredictNeverDrops pins the inference contract: with HiddenDropout
// set, Predict stays deterministic and drop-free -- dropout may only fire
// inside training's gradient accumulation.
func TestPredictNeverDrops(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 32, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)
	n.HiddenDropout = 0.5

	input := []float64{0.3, 0.9}
	first := n.Predict(input)[0]
	for i := 0; i < 10; i++ {
		if got := n.Predict(input)[0]; got != first {
			t.Fatalf("Predict changed between calls with dropout set: %g vs %g", got, first)
		}
	}

	// Sigmoid activations are strictly positive, so a zero would mean a
	// unit was dropped.
	for _, v := range n.Activations[1].Flatten() {
		if v == 0 {
			t.Fatal("Predict dropped a hidden unit")
		}
	}
}

// TestDropoutMasksHiddenUnitsDuringTraining checks the training-side
// mechanics on one sample: some (but not all) hidden sigmoid units read
// exactly zero, which only dropout can produce, and the same sample's
// gradients through those units are zero in both directions.
func TestDropoutMasksHiddenUnitsDuringTraining(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 32, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)
	n.HiddenDropout = 0.5

	sample := DataSet{{Inputs: []float64{0.3, 0.9}, Targets: []float64{1}}}
	weightGrads, _, _ := accumulateBatchGradients(n, sample)

	if n.training {
		t.Fatal("training flag left set after gradient accumulation")
	}

	hidden := n.Activations[1].Flatten()
	dropped := 0
	for j, v := range hidden {
		if v != 0 {
			continue
		}
		dropped++

		// Incoming weights of a dropped unit get no gradient (row j of the
		// first weight matrix)...
		for c := 0; c < weightGrads[0].Columns; c++ {
			if g := weightGrads[0].At(j, c); g != 0 {
				t.Fatalf("dropped unit %d has incoming gradient %g", j, g)
			}
		}
		// ...and neither do its outgoing weights (column j of the second).
		for r := 0; r < weightGrads[1].Rows; r++ {
			if g := weightGrads[1].At(r, j); g != 0 {
				t.Fatalf("dropped unit %d has outgoing gradient %g", j, g)
			}
		}
	}

	if dropped == 0 || dropped == len(hidden) {
		t.Fatalf("dropout at 0.5 dropped %d of %d units; expected some of each", dropped, len(hidden))
	}
}

// TestDropoutPreservesExpectedActivation checks the "inverted" part: over
// many training-mode forward passes, the mean of a surviving-scaled hidden
// activation approximates its dropout-free value.
func TestDropoutPreservesExpectedActivation(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 1},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	input := []float64{0.7}
	n.Predict(input)
	base := n.Activations[1].At(0, 0)

	n.HiddenDropout = 0.5
	n.training = true
	defer func() { n.training = false }()

	const runs = 20000
	sum := 0.0
	for i := 0; i < runs; i++ {
		n.Predict(input)
		sum += n.Activations[1].At(0, 0)
	}

	if mean := sum / runs; math.Abs(mean-base) > 0.02 {
		t.Fatalf("mean training activation %g drifted from clean %g", mean, base)
	}
}

func TestDropoutStillLearnsXOR(t *testing.T) {
	rand.Seed(7)

	n := New(0.05, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 12, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)
	n.HiddenDropout = 0.1

	data := xorData()
	optimizer := Adam(2)
	for i := 0; i < 1500; i++ {
		optimizer(n, data)
	}

	if got := n.Accuracy(data); got != 1 {
		t.Errorf("Accuracy after dropout training = %g, want 1", got)
	}

	mustPanicGoneural(t, "dropout of 1", func() {
		bad := New(0.05, MSE(),
			Layer{Nodes: 2},
			Layer{Nodes: 4, Activator: Sigmoid()},
			Layer{Nodes: 1, Activator: Sigmoid()},
		)
		bad.HiddenDropout = 1
		accumulateBatchGradients(bad, xorData())
	})
}
