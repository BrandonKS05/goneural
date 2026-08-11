package goneural

import (
	"math/rand"
	"testing"
)

func xorData() DataSet {
	return DataSet{
		{Inputs: []float64{1, 0}, Targets: []float64{1}},
		{Inputs: []float64{0, 1}, Targets: []float64{1}},
		{Inputs: []float64{1, 1}, Targets: []float64{0}},
		{Inputs: []float64{0, 0}, Targets: []float64{0}},
	}
}

func testOptimizerLearnsXOR(t *testing.T, optimizer Optimizer, epochs int) {
	t.Helper()

	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	data := xorData()
	firstErr := optimizer(n, data)

	var lastErr float64
	for i := 0; i < epochs; i++ {
		lastErr = optimizer(n, data)
	}

	if lastErr >= firstErr {
		t.Fatalf("expected optimizer to reduce loss, got firstErr=%f lastErr=%f", firstErr, lastErr)
	}

	for _, ds := range data {
		got := n.Predict(ds.Inputs)[0]
		want := ds.Targets[0]
		if diff := got - want; diff > 0.2 || diff < -0.2 {
			t.Errorf("Predict(%v) = %f, want close to %f", ds.Inputs, got, want)
		}
	}
}

func TestMomentumSGDReducesLoss(t *testing.T) {
	rand.Seed(7)
	testOptimizerLearnsXOR(t, MomentumSGD(2, 0.5, 0.9), 800)
}

func TestNesterovSGDReducesLoss(t *testing.T) {
	rand.Seed(7)
	testOptimizerLearnsXOR(t, NesterovSGD(2, 0.5, 0.9), 800)
}

// TestMomentumZeroMatchesPlainDescent pins the degenerate case: with
// Momentum set to 0 the velocity is always exactly the current gradient, so
// the optimizer must still converge like plain mini-batch descent rather
// than stall or blow up.
func TestMomentumZeroMatchesPlainDescent(t *testing.T) {
	rand.Seed(7)
	testOptimizerLearnsXOR(t, MomentumSGD(2, 0.9, 0), 2000)
}
