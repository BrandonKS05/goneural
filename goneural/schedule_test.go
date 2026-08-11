package goneural

import (
	"math"
	"math/rand"
	"testing"
)

func TestStepDecay(t *testing.T) {
	s := StepDecay(1.0, 0.5, 3)

	wants := map[int]float64{1: 1.0, 3: 1.0, 4: 0.5, 6: 0.5, 7: 0.25}
	for epoch, want := range wants {
		if got := s(epoch); math.Abs(got-want) > 1e-12 {
			t.Errorf("StepDecay epoch %d = %g, want %g", epoch, got, want)
		}
	}

	mustPanicGoneural(t, "non-positive interval", func() { StepDecay(1, 0.5, 0) })
}

func TestExponentialDecay(t *testing.T) {
	s := ExponentialDecay(2.0, 0.1)

	if got := s(1); got != 2.0 {
		t.Errorf("epoch 1 = %g, want the initial rate", got)
	}
	if got, want := s(11), 2.0*math.Exp(-1); math.Abs(got-want) > 1e-12 {
		t.Errorf("epoch 11 = %g, want %g", got, want)
	}
}

func TestCosineAnnealing(t *testing.T) {
	s := CosineAnnealing(1.0, 0.1, 10)

	if got := s(1); math.Abs(got-1.0) > 1e-12 {
		t.Errorf("epoch 1 = %g, want the initial rate", got)
	}

	// Halfway through the period the rate is the midpoint of the sweep.
	if got, want := s(6), 0.55; math.Abs(got-want) > 1e-12 {
		t.Errorf("epoch 6 = %g, want %g", got, want)
	}

	// The restart at the period boundary jumps back to the initial rate.
	if got := s(11); math.Abs(got-1.0) > 1e-12 {
		t.Errorf("epoch 11 (restart) = %g, want the initial rate", got)
	}

	// The schedule never leaves [min, initial].
	for epoch := 1; epoch <= 30; epoch++ {
		if lr := s(epoch); lr < 0.1-1e-12 || lr > 1.0+1e-12 {
			t.Fatalf("epoch %d = %g, outside [0.1, 1.0]", epoch, lr)
		}
	}
}

func TestWithScheduleDrivesNetworkLearningRate(t *testing.T) {
	var seen []float64
	fake := func(n *NeuralNetwork, dataSet DataSet) float64 {
		seen = append(seen, n.LearningRate)
		return 0
	}

	n := New(0.9, MSE(),
		Layer{Nodes: 1},
		Layer{Nodes: 1, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	opt := WithSchedule(fake, StepDecay(1.0, 0.5, 1))
	for i := 0; i < 3; i++ {
		opt(n, DataSet{})
	}

	wants := []float64{1.0, 0.5, 0.25}
	for i, want := range wants {
		if math.Abs(seen[i]-want) > 1e-12 {
			t.Errorf("epoch %d ran with LR %g, want %g", i+1, seen[i], want)
		}
	}
}

func TestWithScheduleFuncDrivesExternalLearningRate(t *testing.T) {
	rand.Seed(7)

	// Decay gently: XOR needs a workable learning rate for the whole run,
	// so the point here is only that the schedule visibly drives the field.
	o := NewMomentumOptimizer(2, 0.5, 0.9)
	opt := WithScheduleFunc(o.Optimize, ExponentialDecay(0.5, 0.0005),
		func(lr float64) { o.LearningRate = lr })

	testOptimizerLearnsXOR(t, opt, 800)

	if o.LearningRate >= 0.5 {
		t.Errorf("expected the schedule to have decayed the optimizer's LR, still %g", o.LearningRate)
	}
}

func mustPanicGoneural(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: expected panic, got none", name)
		}
	}()
	f()
}
