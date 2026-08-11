package goneural

import (
	"math"
	"math/rand"
	"testing"
)

func TestLookaheadReducesLoss(t *testing.T) {
	rand.Seed(7)
	testOptimizerLearnsXOR(t, Lookahead(Adam(2), 5, 0.5), 500)
}

// TestLookaheadInterpolatesTowardFastWeights drives the wrapper with a fake
// inner optimizer that shifts every weight by a fixed amount per call, so
// the slow-weight update is exactly predictable: after k fast steps of d,
// the sync must land the network on start + alpha*k*d.
func TestLookaheadInterpolatesTowardFastWeights(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 1},
		Layer{Nodes: 2, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	const (
		d     = 0.1
		k     = 4
		alpha = 0.5
	)

	start := n.Weights[0].At(0, 0)

	fake := func(net *NeuralNetwork, dataSet DataSet) float64 {
		for l := range net.Weights {
			net.Weights[l] = net.Weights[l].Add(d)
		}
		return 0
	}

	opt := Lookahead(fake, k, alpha)

	// One invocation short of the sync: still pure fast weights.
	for i := 0; i < k-1; i++ {
		opt(n, DataSet{})
	}
	if got, want := n.Weights[0].At(0, 0), start+float64(k-1)*d; math.Abs(got-want) > 1e-12 {
		t.Fatalf("before sync: weight = %g, want fast value %g", got, want)
	}

	// The k-th invocation triggers the sync back onto the slow weights.
	opt(n, DataSet{})
	if got, want := n.Weights[0].At(0, 0), start+alpha*float64(k)*d; math.Abs(got-want) > 1e-12 {
		t.Fatalf("after sync: weight = %g, want interpolated value %g", got, want)
	}

	mustPanicGoneural(t, "bad sync period", func() { Lookahead(fake, 0, 0.5) })
	mustPanicGoneural(t, "bad alpha", func() { Lookahead(fake, 5, 1.5) })
}
