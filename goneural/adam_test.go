package goneural

import (
	"math/rand"
	"testing"
)

func TestAdamReducesLoss(t *testing.T) {
	rand.Seed(7)

	n := New(0.05, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	data := DataSet{
		{Inputs: []float64{1, 0}, Targets: []float64{1}},
		{Inputs: []float64{0, 1}, Targets: []float64{1}},
		{Inputs: []float64{1, 1}, Targets: []float64{0}},
		{Inputs: []float64{0, 0}, Targets: []float64{0}},
	}

	optimizer := Adam(2)
	firstErr := optimizer(n, data)

	var lastErr float64
	for i := 0; i < 500; i++ {
		lastErr = optimizer(n, data)
	}

	if lastErr >= firstErr {
		t.Fatalf("expected Adam to reduce loss, got firstErr=%f lastErr=%f", firstErr, lastErr)
	}

	for _, ds := range data {
		got := n.Predict(ds.Inputs)[0]
		want := ds.Targets[0]
		if diff := got - want; diff > 0.2 || diff < -0.2 {
			t.Errorf("Predict(%v) = %f, want close to %f", ds.Inputs, got, want)
		}
	}
}
