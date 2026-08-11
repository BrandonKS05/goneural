package goneural

import (
	"math"
	"math/rand"
	"testing"
)

// TestWithWeightDecayShrinksGeometrically drives the wrapper with an inert
// inner optimizer, so the only thing moving the weights is the decay: after
// k epochs every weight must be exactly w0 * (1-rate)^k.
func TestWithWeightDecayShrinksGeometrically(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 3, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	const rate = 0.1
	const epochs = 5

	w0 := n.Weights[0].At(0, 0)
	b0 := n.Biases[0].At(0, 0)

	inert := func(net *NeuralNetwork, dataSet DataSet) float64 { return 0 }
	opt := WithWeightDecay(inert, rate)
	for i := 0; i < epochs; i++ {
		opt(n, DataSet{})
	}

	want := w0 * math.Pow(1-rate, epochs)
	if got := n.Weights[0].At(0, 0); math.Abs(got-want) > 1e-12 {
		t.Errorf("weight after %d epochs = %g, want %g", epochs, got, want)
	}

	// Biases must not decay.
	if got := n.Biases[0].At(0, 0); got != b0 {
		t.Errorf("bias decayed from %g to %g", b0, got)
	}

	mustPanicGoneural(t, "negative rate", func() { WithWeightDecay(inert, -0.1) })
	mustPanicGoneural(t, "rate of 1", func() { WithWeightDecay(inert, 1) })
}

func TestWithWeightDecayStillLearns(t *testing.T) {
	rand.Seed(7)
	testOptimizerLearnsXOR(t, WithWeightDecay(Adam(2), 1e-4), 500)
}
