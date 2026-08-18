package goneural

import (
	"math"
	"math/rand"
	"testing"
)

// fixedStepOptimizer is a fake inner optimizer that shifts every weight and
// bias by a known amount per call, making the running average exactly
// predictable.
func fixedStepOptimizer(d float64) Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		for l := range n.Weights {
			n.Weights[l] = n.Weights[l].Add(d)
			n.Biases[l] = n.Biases[l].Add(d)
		}
		return 0
	}
}

func averagingNetwork() *NeuralNetwork {
	return New(0.1, MSE(),
		Layer{Nodes: 1},
		Layer{Nodes: 2, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)
}

// TestSWATakesEqualWeightMean walks the weights along a straight line and
// checks the average lands on the arithmetic mean of the visited points.
func TestSWATakesEqualWeightMean(t *testing.T) {
	rand.Seed(7)

	n := averagingNetwork()
	start := n.Weights[0].At(0, 0)

	const (
		d     = 0.25
		steps = 4
	)

	avg := SWA(fixedStepOptimizer(d), 0)
	for i := 0; i < steps; i++ {
		avg.Optimize(n, DataSet{})
	}

	if avg.Count() != steps {
		t.Fatalf("Count() = %d, want %d", avg.Count(), steps)
	}

	// The visited weights are start+d, start+2d, ..., start+steps*d.
	want := start + d*float64(steps+1)/2
	if !avg.Apply(n) {
		t.Fatal("Apply reported no average to apply")
	}
	if got := n.Weights[0].At(0, 0); math.Abs(got-want) > 1e-12 {
		t.Fatalf("averaged weight = %g, want mean %g", got, want)
	}
}

// TestEMADecaysOldSnapshots pins the exponential recurrence against a
// hand-rolled scalar version of the same update.
func TestEMADecaysOldSnapshots(t *testing.T) {
	rand.Seed(7)

	n := averagingNetwork()
	start := n.Biases[0].At(0, 0)

	const (
		d     = 0.5
		decay = 0.8
		steps = 5
	)

	avg := EMA(fixedStepOptimizer(d), decay)

	want := start + d // seeded from the first snapshot
	for i := 0; i < steps; i++ {
		avg.Optimize(n, DataSet{})
		if i > 0 {
			want = decay*want + (1-decay)*(start+d*float64(i+1))
		}
	}

	avg.Apply(n)
	if got := n.Biases[0].At(0, 0); math.Abs(got-want) > 1e-12 {
		t.Fatalf("EMA bias = %g, want %g", got, want)
	}
}

// TestWeightAveragerStartAfterSkipsWarmup checks the leading invocations
// stay out of the mean entirely.
func TestWeightAveragerStartAfterSkipsWarmup(t *testing.T) {
	rand.Seed(7)

	n := averagingNetwork()
	start := n.Weights[0].At(0, 0)

	const d = 1.0

	avg := SWA(fixedStepOptimizer(d), 2)

	// Nothing accumulated yet, so there is nothing to apply.
	avg.Optimize(n, DataSet{})
	avg.Optimize(n, DataSet{})
	if avg.Apply(n) {
		t.Fatal("Apply succeeded before any snapshot was taken")
	}

	// The third and fourth invocations land on start+3d and start+4d.
	avg.Optimize(n, DataSet{})
	avg.Optimize(n, DataSet{})

	avg.Apply(n)
	if got, want := n.Weights[0].At(0, 0), start+3.5*d; math.Abs(got-want) > 1e-12 {
		t.Fatalf("averaged weight = %g, want %g", got, want)
	}
}

// TestWeightAveragerLeavesTrainingUntouched makes sure the wrapper is
// transparent: the live weights must be exactly what the inner optimizer
// produced until Apply is called.
func TestWeightAveragerLeavesTrainingUntouched(t *testing.T) {
	rand.Seed(7)

	plain := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)
	wrapped := plain.Copy()

	data := xorData()
	inner := Adam(2)
	avg := EMA(Adam(2), 0.9)

	for i := 0; i < 20; i++ {
		data.Shuffle()
		inner(plain, data)
		avg.Optimize(wrapped, data)
	}

	for l := range plain.Weights {
		for i := 0; i < plain.Weights[l].Rows; i++ {
			for j := 0; j < plain.Weights[l].Columns; j++ {
				if got, want := wrapped.Weights[l].At(i, j), plain.Weights[l].At(i, j); math.Abs(got-want) > 1e-12 {
					t.Fatalf("layer %d weight (%d, %d) = %g, want untouched %g", l, i, j, got, want)
				}
			}
		}
	}
}

func TestWeightAveragerReducesLoss(t *testing.T) {
	rand.Seed(7)
	testOptimizerLearnsXOR(t, EMA(Adam(2), 0.9).Optimize, 500)
}

func TestWeightAveragerRejectsBadParameters(t *testing.T) {
	inner := fixedStepOptimizer(0)

	mustPanicGoneural(t, "negative decay", func() { NewWeightAverager(inner, -0.1, 0) })
	mustPanicGoneural(t, "decay of one", func() { NewWeightAverager(inner, 1, 0) })
	mustPanicGoneural(t, "negative startAfter", func() { NewWeightAverager(inner, 0, -1) })
	mustPanicGoneural(t, "zero EMA decay", func() { EMA(inner, 0) })

	mustPanicGoneural(t, "mismatched network", func() {
		n := averagingNetwork()
		avg := SWA(fixedStepOptimizer(0.1), 0)
		avg.Optimize(n, DataSet{})

		avg.Apply(New(0.1, MSE(),
			Layer{Nodes: 1},
			Layer{Nodes: 2, Activator: Sigmoid()},
			Layer{Nodes: 2, Activator: Sigmoid()},
			Layer{Nodes: 1, Activator: Sigmoid()},
		))
	})
}
