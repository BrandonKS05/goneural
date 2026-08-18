package goneural

import (
	"math"
	"math/rand"
	"testing"
)

func lrFinderNetwork() *NeuralNetwork {
	return New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)
}

// TestLRFinderTracesTheCurve runs a real sweep on XOR and checks the shape
// the range test is supposed to produce: rates ramp geometrically, the loss
// falls below where it started, and it turns back up before the top of the
// range -- so the suggested rate sits on the descent, short of the minimum.
func TestLRFinderTracesTheCurve(t *testing.T) {
	rand.Seed(7)

	n := lrFinderNetwork()

	o := NewMomentumOptimizer(2, 1e-6, 0.9)
	finder := NewLRFinder(o.Optimize, func(lr float64) { o.LearningRate = lr }, 1e-6, 100, 25)
	finder.EpochsPerStep = 5

	sweep := finder.Run(n, xorData())
	if len(sweep) < 2 {
		t.Fatalf("sweep has %d probes, want a curve", len(sweep))
	}

	for i := 1; i < len(sweep); i++ {
		if sweep[i].LearningRate <= sweep[i-1].LearningRate {
			t.Fatalf("probe %d has rate %g, not above the previous %g",
				i, sweep[i].LearningRate, sweep[i-1].LearningRate)
		}
	}

	best, ok := sweep.Best()
	if !ok {
		t.Fatal("Best found no usable probe")
	}
	if best.Loss >= sweep[0].Loss {
		t.Errorf("best loss %g is no better than the smallest rate's %g", best.Loss, sweep[0].Loss)
	}
	if last := sweep[len(sweep)-1]; last.Loss <= best.Loss {
		t.Errorf("loss at the top of the range is %g, want it back above the best %g", last.Loss, best.Loss)
	}

	suggested, ok := sweep.Suggest(finder.Smoothing)
	if !ok {
		t.Fatal("Suggest found no descending stretch")
	}
	if suggested < finder.Min || suggested > finder.Max {
		t.Errorf("suggested rate %g outside the swept range [%g, %g]", suggested, finder.Min, finder.Max)
	}
	if suggested > best.LearningRate {
		t.Errorf("suggested rate %g is past the loss minimum at %g", suggested, best.LearningRate)
	}
}

// TestLRFinderLeavesNetworkUntouched pins that the sweep trains a copy --
// it deliberately pushes the weights to divergence, which would be a
// disaster to do in place.
func TestLRFinderLeavesNetworkUntouched(t *testing.T) {
	rand.Seed(7)

	n := lrFinderNetwork()
	before := n.Weights[0].Copy()

	o := NewMomentumOptimizer(2, 1e-6, 0.9)
	NewLRFinder(o.Optimize, func(lr float64) { o.LearningRate = lr }, 1e-6, 100, 20).
		Run(n, xorData())

	for i := 0; i < before.Rows; i++ {
		for j := 0; j < before.Columns; j++ {
			if got, want := n.Weights[0].At(i, j), before.At(i, j); got != want {
				t.Fatalf("weight (%d, %d) = %g, want the untouched %g", i, j, got, want)
			}
		}
	}
}

// TestLRFinderRunsEveryEpochPerStep checks the per-probe training budget
// reaches the optimizer.
func TestLRFinderRunsEveryEpochPerStep(t *testing.T) {
	calls := 0
	fake := func(net *NeuralNetwork, dataSet DataSet) float64 {
		calls++
		return 0
	}

	finder := NewLRFinder(fake, func(float64) {}, 1e-5, 1, 6)
	finder.EpochsPerStep = 3
	finder.Run(lrFinderNetwork(), xorData())

	if want := 6 * 3; calls != want {
		t.Errorf("optimizer ran %d times, want %d", calls, want)
	}
}

// sweepOf builds a curve at one probe per decade from 1e-5 up, so the
// expected answers of Best and Suggest can be written down by hand.
func sweepOf(losses ...float64) LRSweep {
	sweep := make(LRSweep, len(losses))
	for i, loss := range losses {
		sweep[i] = LRProbe{LearningRate: 1e-5 * math.Pow(10, float64(i)), Loss: loss}
	}
	return sweep
}

// TestSuggestPicksTheSteepestDescent hands Suggest a curve whose steepest
// decade is known by construction.
func TestSuggestPicksTheSteepestDescent(t *testing.T) {
	// The drop from probe 2 to probe 3 is the steepest, so the rate to
	// train at is probe 2's -- not probe 4's, where the loss bottoms out
	// and the optimizer is already near the edge.
	sweep := sweepOf(1.0, 0.99, 0.95, 0.30, 0.28, 5.0)

	got, ok := sweep.Suggest(0)
	if !ok {
		t.Fatal("Suggest found no descending stretch")
	}
	if want := sweep[2].LearningRate; math.Abs(got-want) > 1e-12 {
		t.Errorf("Suggest = %g, want the steepest decade's %g", got, want)
	}

	best, ok := sweep.Best()
	if !ok {
		t.Fatal("Best found no usable probe")
	}
	if want := sweep[4].LearningRate; best.LearningRate != want {
		t.Errorf("Best = %g, want the minimum at %g", best.LearningRate, want)
	}
}

// TestSuggestNormalizesByDecade guards the log-space gradient: probe 1
// covers 9e-5 of raw learning rate and probe 4 covers 0.09, so measuring
// the slope against the raw rate would rank a negligible early wobble
// above the real collapse a thousandfold further along.
func TestSuggestNormalizesByDecade(t *testing.T) {
	sweep := sweepOf(1.0, 0.99, 0.99, 0.99, 0.30, 0.30)

	got, ok := sweep.Suggest(0)
	if !ok {
		t.Fatal("Suggest found no descending stretch")
	}
	if want := sweep[3].LearningRate; math.Abs(got-want) > 1e-12 {
		t.Errorf("Suggest = %g, want the rate before the real drop, %g", got, want)
	}
}

// TestSuggestSkipsNonFiniteProbes checks a diverged tail cannot poison the
// smoothing or the answer.
func TestSuggestSkipsNonFiniteProbes(t *testing.T) {
	sweep := sweepOf(1.0, 0.9, 0.2, math.Inf(1), math.NaN())

	got, ok := sweep.Suggest(0)
	if !ok {
		t.Fatal("Suggest found no descending stretch")
	}
	if want := sweep[1].LearningRate; math.Abs(got-want) > 1e-12 {
		t.Errorf("Suggest = %g, want %g", got, want)
	}

	best, ok := sweep.Best()
	if !ok {
		t.Fatal("Best found no usable probe")
	}
	if want := sweep[2].LearningRate; best.LearningRate != want {
		t.Errorf("Best = %g, want the finite minimum at %g", best.LearningRate, want)
	}
}

// TestSuggestReportsNoDescent covers the "swept range was all cliff" case.
func TestSuggestReportsNoDescent(t *testing.T) {
	if got, ok := sweepOf(0.1, 0.5, 2.0, 9.0).Suggest(0); ok {
		t.Errorf("Suggest = %g on a monotonically rising curve, want no answer", got)
	}

	if _, ok := (LRSweep{}).Best(); ok {
		t.Error("Best reported a probe on an empty sweep")
	}
	if _, ok := (LRSweep{}).Suggest(0); ok {
		t.Error("Suggest reported a rate on an empty sweep")
	}
	if _, ok := (LRSweep{{LearningRate: 1, Loss: math.NaN()}}).Best(); ok {
		t.Error("Best reported a probe on an all-NaN sweep")
	}
}

// TestLRFinderStopsOnDivergence checks both early exits: the ratio test on
// a loss that climbs past the cutoff, and a loss that goes non-finite
// outright. The fake optimizer wrecks the weights on cue rather than
// reporting a number, since the sweep reads the loss off the network.
func TestLRFinderStopsOnDivergence(t *testing.T) {
	data := xorData()

	// An unsquashed output layer, so wrecked weights actually show up as a
	// wrecked loss; a sigmoid output would merely saturate.
	network := func() *NeuralNetwork {
		return New(0.1, MSE(),
			Layer{Nodes: 2},
			Layer{Nodes: 4, Activator: Sigmoid()},
			Layer{Nodes: 1, Activator: Identity()},
		)
	}

	wreckAt := func(step int, scale float64) Optimizer {
		called := 0
		return func(n *NeuralNetwork, dataSet DataSet) float64 {
			if called == step {
				for l := range n.Weights {
					n.Weights[l] = n.Weights[l].Scale(scale)
				}
			}
			called++
			return 0
		}
	}

	finder := NewLRFinder(wreckAt(2, 1e6), func(float64) {}, 1e-5, 1, 8)
	finder.Smoothing = 0
	if got := len(finder.Run(network(), data)); got != 3 {
		t.Errorf("sweep ran %d probes, want it to stop at 3 once the loss blew up", got)
	}

	// The same blow-up must be ridden out when the check is disabled.
	finder = NewLRFinder(wreckAt(2, 1e6), func(float64) {}, 1e-5, 1, 8)
	finder.Smoothing = 0
	finder.DivergeFactor = 0
	if got := len(finder.Run(network(), data)); got != 8 {
		t.Errorf("sweep ran %d probes with the divergence check off, want all 8", got)
	}

	// Weights driven to infinity make the predictions NaN, which stops the
	// sweep regardless of DivergeFactor.
	finder = NewLRFinder(wreckAt(1, math.Inf(1)), func(float64) {}, 1e-5, 1, 8)
	finder.DivergeFactor = 0
	if got := len(finder.Run(network(), data)); got != 2 {
		t.Errorf("sweep ran %d probes, want it to stop at 2 on a non-finite loss", got)
	}
}

func TestLRFinderRejectsBadParameters(t *testing.T) {
	noop := func(net *NeuralNetwork, dataSet DataSet) float64 { return 0 }
	setLR := func(float64) {}

	mustPanicGoneural(t, "zero min", func() { NewLRFinder(noop, setLR, 0, 1, 10) })
	mustPanicGoneural(t, "max below min", func() { NewLRFinder(noop, setLR, 1, 0.1, 10) })
	mustPanicGoneural(t, "one step", func() { NewLRFinder(noop, setLR, 1e-5, 1, 1) })
	mustPanicGoneural(t, "missing hook", func() { NewLRFinder(noop, nil, 1e-5, 1, 10) })

	mustPanicGoneural(t, "empty data set", func() {
		NewLRFinder(noop, setLR, 1e-5, 1, 10).Run(lrFinderNetwork(), DataSet{})
	})
	mustPanicGoneural(t, "bad smoothing", func() {
		f := NewLRFinder(noop, setLR, 1e-5, 1, 10)
		f.Smoothing = 1
		f.Run(lrFinderNetwork(), xorData())
	})
	mustPanicGoneural(t, "bad smoothing", func() { LRSweep{}.Suggest(1) })
}
