package goneural

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
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

func TestPrecisionRecallF1(t *testing.T) {
	// Hand-built binary confusion matrix: 5 true 0s predicted 0, 1 true 0
	// predicted 1, 2 true 1s predicted 0, 4 true 1s predicted 1.
	confusion := matrix.New(2, 2, [][]float64{{5, 1}, {2, 4}})

	if got, want := Precision(confusion, 0), 5.0/7.0; math.Abs(got-want) > 1e-12 {
		t.Errorf("Precision(0) = %g, want %g", got, want)
	}
	if got, want := Recall(confusion, 0), 5.0/6.0; math.Abs(got-want) > 1e-12 {
		t.Errorf("Recall(0) = %g, want %g", got, want)
	}

	p, r := 4.0/5.0, 4.0/6.0
	if got, want := F1Score(confusion, 1), 2*p*r/(p+r); math.Abs(got-want) > 1e-12 {
		t.Errorf("F1Score(1) = %g, want %g", got, want)
	}

	// Degenerate cases collapse to 0 instead of dividing by zero: class 1
	// is never predicted (empty column) and never occurs (empty row).
	never := matrix.New(2, 2, [][]float64{{3, 0}, {0, 0}})
	if got := Precision(never, 1); got != 0 {
		t.Errorf("Precision of never-predicted class = %g, want 0", got)
	}
	if got := Recall(never, 1); got != 0 {
		t.Errorf("Recall of absent class = %g, want 0", got)
	}
	if got := F1Score(never, 1); got != 0 {
		t.Errorf("F1 of absent class = %g, want 0", got)
	}
}

func TestConfusionMatrixOnTrainedXOR(t *testing.T) {
	rand.Seed(7)

	n := New(0.05, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	data := xorData()
	optimizer := Adam(2)
	for i := 0; i < 500; i++ {
		optimizer(n, data)
	}

	confusion := n.ConfusionMatrix(data)

	// A perfectly learned XOR puts both 0-samples and both 1-samples on
	// the diagonal of the binary 2x2 matrix.
	want := matrix.New(2, 2, [][]float64{{2, 0}, {0, 2}})
	if !confusion.Equal(want) {
		t.Fatalf("confusion matrix = %v, want %v", confusion, want)
	}

	if got := F1Score(confusion, 1); got != 1 {
		t.Errorf("F1 on perfect classifier = %g, want 1", got)
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
