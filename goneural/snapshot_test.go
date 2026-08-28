package goneural

import (
	"math"
	"math/rand"
	"testing"
)

// TestSnapshotsAreTakenOnSchedule pins the interval, the warm-up offset,
// and that the copies really are independent of the live network.
func TestSnapshotsAreTakenOnSchedule(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 1},
		Layer{Nodes: 2, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	const (
		d      = 0.5
		every  = 3
		warmUp = 2
	)

	start := n.Weights[0].At(0, 0)

	o := NewSnapshotOptimizer(fixedStepOptimizer(d), every)
	o.WarmUp = warmUp

	for i := 0; i < 11; i++ {
		o.Optimize(n, DataSet{})
	}

	// Invocations 5, 8 and 11 are the ones warm-up + interval select.
	if got := o.Count(); got != 3 {
		t.Fatalf("took %d snapshots over 11 invocations, want 3", got)
	}

	for i, want := range []float64{5, 8, 11} {
		member := o.Ensemble()[i]
		if got := member.Weights[0].At(0, 0); math.Abs(got-(start+want*d)) > 1e-12 {
			t.Errorf("snapshot %d holds weight %g, want the value after %v invocations", i, got, want)
		}
	}

	// The live network moves on; the snapshots must not follow it.
	before := o.Ensemble()[0].Weights[0].At(0, 0)
	o.Optimize(n, DataSet{})
	if after := o.Ensemble()[0].Weights[0].At(0, 0); after != before {
		t.Errorf("snapshot changed from %g to %g when training continued", before, after)
	}
}

// TestSnapshotLimitKeepsTheLatest checks the eviction end: later cycles
// are the better ones, so the oldest is what should go.
func TestSnapshotLimitKeepsTheLatest(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 1},
		Layer{Nodes: 2, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)
	start := n.Weights[0].At(0, 0)

	o := NewSnapshotOptimizer(fixedStepOptimizer(1), 1)
	o.Limit = 2

	for i := 0; i < 5; i++ {
		o.Optimize(n, DataSet{})
	}

	if got := o.Count(); got != 2 {
		t.Fatalf("kept %d snapshots, want the limit of 2", got)
	}
	for i, want := range []float64{4, 5} {
		if got := o.Ensemble()[i].Weights[0].At(0, 0); math.Abs(got-(start+want)) > 1e-12 {
			t.Errorf("snapshot %d holds %g, want the one from invocation %v", i, got, want)
		}
	}
}

// TestSnapshotEnsembleBeatsItsMembers is the point of the technique: the
// committee collected along a single cyclic run should classify at least
// as well as its typical member, at no extra training cost.
func TestSnapshotEnsembleBeatsItsMembers(t *testing.T) {
	rand.Seed(3)

	data := twoClusterData(120)

	n := New(0.1, CrossEntropy(),
		Layer{Nodes: 2},
		Layer{Nodes: 8, Activator: Tanh()},
		Layer{Nodes: 2, Activator: Softmax()},
	)

	const period = 20

	inner := NewNadamOptimizer(8, 0.05)
	scheduled := WithScheduleFunc(inner.Optimize,
		CosineAnnealing(0.05, 0.0005, period),
		func(lr float64) { inner.LearningRate = lr })

	snapshots := NewSnapshotOptimizer(scheduled, period)
	snapshots.WarmUp = period // skip the first cycle

	n.Train(snapshots.Optimize, data, 6*period)

	ensemble := snapshots.Ensemble()
	if len(ensemble) < 3 {
		t.Fatalf("collected %d snapshots over 6 cycles, want at least 3", len(ensemble))
	}

	// Different cycles must have reached different weights, or averaging
	// them buys nothing at all.
	if ensemble[0].Weights[0].At(0, 0) == ensemble[len(ensemble)-1].Weights[0].At(0, 0) {
		t.Error("first and last snapshots share a weight exactly, so the cycles never moved")
	}

	mean := 0.0
	for _, member := range ensemble {
		mean += member.Accuracy(data)
	}
	mean /= float64(len(ensemble))

	if got := ensemble.Accuracy(data); got < mean {
		t.Errorf("ensemble accuracy %g, below its members' mean of %g", got, mean)
	}
}

// twoClusterData returns a small two-class problem: overlapping Gaussian
// blobs, easy enough to learn quickly but not separable by accident.
func twoClusterData(n int) DataSet {
	data := make(DataSet, n)
	for i := range data {
		class := i % 2
		center := -1.0
		if class == 1 {
			center = 1
		}

		data[i] = DataSample{
			Inputs: []float64{
				center + rand.NormFloat64()*0.7,
				center + rand.NormFloat64()*0.7,
			},
			Targets: OneHot(2, class),
		}
	}
	return data
}

func TestSnapshotRejectsBadParameters(t *testing.T) {
	inner := fixedStepOptimizer(0)

	mustPanicGoneural(t, "zero interval", func() { NewSnapshotOptimizer(inner, 0) })
	mustPanicGoneural(t, "negative warm-up", func() {
		o := NewSnapshotOptimizer(inner, 2)
		o.WarmUp = -1
		o.Optimize(New(0.1, MSE(),
			Layer{Nodes: 1},
			Layer{Nodes: 2, Activator: Sigmoid()},
			Layer{Nodes: 1, Activator: Sigmoid()},
		), DataSet{})
	})

	empty := NewSnapshotOptimizer(inner, 2)
	if got := empty.Count(); got != 0 {
		t.Errorf("a fresh optimizer holds %d snapshots, want 0", got)
	}
	if got := empty.Ensemble(); len(got) != 0 {
		t.Errorf("a fresh optimizer returned %d members, want none", len(got))
	}
}
