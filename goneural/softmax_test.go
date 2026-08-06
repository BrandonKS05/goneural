package goneural

import (
	"math/rand"
	"testing"
)

func threeClassData() DataSet {
	return DataSet{
		{Inputs: []float64{0, 0}, Targets: []float64{1, 0, 0}},
		{Inputs: []float64{0.2, -0.1}, Targets: []float64{1, 0, 0}},
		{Inputs: []float64{-0.1, 0.2}, Targets: []float64{1, 0, 0}},
		{Inputs: []float64{5, 5}, Targets: []float64{0, 1, 0}},
		{Inputs: []float64{5.2, 4.9}, Targets: []float64{0, 1, 0}},
		{Inputs: []float64{4.8, 5.1}, Targets: []float64{0, 1, 0}},
		{Inputs: []float64{-5, 5}, Targets: []float64{0, 0, 1}},
		{Inputs: []float64{-5.1, 4.8}, Targets: []float64{0, 0, 1}},
		{Inputs: []float64{-4.9, 5.2}, Targets: []float64{0, 0, 1}},
	}
}

func argmax(v []float64) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

func TestSoftmaxCrossEntropyClassifies(t *testing.T) {
	rand.Seed(11)

	n := New(0.05, CrossEntropy(),
		Layer{Nodes: 2},
		Layer{Nodes: 8, Activator: Sigmoid()},
		Layer{Nodes: 3, Activator: Softmax()},
	)

	data := threeClassData()
	optimizer := Adam(len(data))
	for i := 0; i < 400; i++ {
		optimizer(n, data)
	}

	for _, ds := range data {
		out := n.Predict(ds.Inputs)

		sum := 0.0
		for _, p := range out {
			sum += p
		}
		if diff := sum - 1; diff > 1e-6 || diff < -1e-6 {
			t.Errorf("Predict(%v) = %v, softmax output should sum to 1, got %f", ds.Inputs, out, sum)
		}

		if got, want := argmax(out), argmax(ds.Targets); got != want {
			t.Errorf("Predict(%v) = %v, predicted class %d, want class %d", ds.Inputs, out, got, want)
		}
	}
}

func TestSoftmaxWithoutCrossEntropyPanics(t *testing.T) {
	rand.Seed(1)

	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 3, Activator: Softmax()},
	)

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected Softmax output paired with a non-CrossEntropy loss to panic")
		}
	}()

	n.Train(SGD(), threeClassData(), 1)
}
