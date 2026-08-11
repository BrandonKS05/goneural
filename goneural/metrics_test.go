package goneural

import (
	"math/rand"
	"testing"
)

func TestArgMax(t *testing.T) {
	if got := ArgMax([]float64{0.1, 0.7, 0.2}); got != 1 {
		t.Errorf("ArgMax = %d, want 1", got)
	}
	if got := ArgMax([]float64{3}); got != 0 {
		t.Errorf("ArgMax of single element = %d, want 0", got)
	}
	// First index wins ties.
	if got := ArgMax([]float64{0.5, 0.5}); got != 0 {
		t.Errorf("ArgMax tie = %d, want 0", got)
	}

	mustPanicGoneural(t, "empty slice", func() { ArgMax(nil) })
}

func TestOneHot(t *testing.T) {
	got := OneHot(4, 2)
	want := []float64{0, 0, 1, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("OneHot(4, 2) = %v, want %v", got, want)
		}
	}

	if idx := ArgMax(OneHot(10, 7)); idx != 7 {
		t.Errorf("ArgMax(OneHot(10, 7)) = %d, want 7", idx)
	}

	mustPanicGoneural(t, "index out of range", func() { OneHot(3, 3) })
	mustPanicGoneural(t, "negative index", func() { OneHot(3, -1) })
}

func TestAccuracyBinary(t *testing.T) {
	rand.Seed(7)

	n := New(0.05, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	data := xorData()

	if got := n.Accuracy(DataSet{}); got != 0 {
		t.Errorf("Accuracy of empty data set = %g, want 0", got)
	}

	optimizer := Adam(2)
	for i := 0; i < 500; i++ {
		optimizer(n, data)
	}

	if got := n.Accuracy(data); got != 1 {
		t.Errorf("Accuracy after training = %g, want 1", got)
	}
}

func TestAccuracyMultiClass(t *testing.T) {
	rand.Seed(7)

	// A one-hot toy problem: classify which quadrant-ish corner the two
	// binary inputs form, so ArgMax-vs-ArgMax comparison is exercised.
	n := New(0.05, CrossEntropy(),
		Layer{Nodes: 2},
		Layer{Nodes: 8, Activator: Sigmoid()},
		Layer{Nodes: 4, Activator: Softmax()},
	)

	data := DataSet{
		{Inputs: []float64{0, 0}, Targets: OneHot(4, 0)},
		{Inputs: []float64{0, 1}, Targets: OneHot(4, 1)},
		{Inputs: []float64{1, 0}, Targets: OneHot(4, 2)},
		{Inputs: []float64{1, 1}, Targets: OneHot(4, 3)},
	}

	optimizer := Adam(2)
	for i := 0; i < 500; i++ {
		optimizer(n, data)
	}

	if got := n.Accuracy(data); got != 1 {
		t.Errorf("Accuracy after training = %g, want 1", got)
	}
}
