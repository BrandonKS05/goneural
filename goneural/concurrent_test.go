package goneural

import (
	"math"
	"math/rand"
	"testing"
)

// TestConcurrentMBGDMatchesMBGD checks that parallelizing the per-sample
// gradient computation doesn't change the math: given the same starting
// weights, ConcurrentMBGD should take the exact same step as sequential
// MBGD. Run with -race to also confirm the goroutines aren't touching
// shared state unsafely.
func TestConcurrentMBGDMatchesMBGD(t *testing.T) {
	rand.Seed(3)

	build := func() *NeuralNetwork {
		return New(0.2, MSE(),
			Layer{Nodes: 3},
			Layer{Nodes: 6, Activator: Sigmoid()},
			Layer{Nodes: 4, Activator: Sigmoid()},
			Layer{Nodes: 2, Activator: Sigmoid()},
		)
	}

	data := DataSet{
		{Inputs: []float64{0.1, 0.4, -0.2}, Targets: []float64{1, 0}},
		{Inputs: []float64{-0.5, 0.2, 0.9}, Targets: []float64{0, 1}},
		{Inputs: []float64{0.3, -0.1, 0.6}, Targets: []float64{1, 1}},
		{Inputs: []float64{0.7, 0.8, -0.3}, Targets: []float64{0, 0}},
		{Inputs: []float64{-0.2, -0.4, 0.1}, Targets: []float64{1, 0}},
	}

	seq := build()
	rand.Seed(3)
	par := build()

	MBGD(2)(seq, data)
	ConcurrentMBGD(2)(par, data)

	for l := range seq.Weights {
		got := par.Weights[l].Flatten()
		want := seq.Weights[l].Flatten()
		for k := range want {
			if diff := math.Abs(got[k] - want[k]); diff > 1e-12 {
				t.Fatalf("layer %d weight[%d]: ConcurrentMBGD=%g MBGD=%g (diff %g)", l, k, got[k], want[k], diff)
			}
		}
	}
}
