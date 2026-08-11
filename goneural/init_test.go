package goneural

import (
	"math"
	"math/rand"
	"testing"
)

func TestInitXavierBoundsAndBiases(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 8},
		Layer{Nodes: 3, Activator: Tanh()},
		Layer{Nodes: 5, Activator: Sigmoid()},
	)
	n.InitXavier()

	for l := range n.Weights {
		fanIn := float64(n.Layers[l].Nodes)
		fanOut := float64(n.Layers[l+1].Nodes)
		limit := math.Sqrt(6 / (fanIn + fanOut))

		if max := n.Weights[l].Map(func(v float64, x, y int) float64 {
			return math.Abs(v)
		}).Max(); max > limit {
			t.Errorf("layer %d: |weight| %g exceeds Xavier limit %g", l, max, limit)
		}

		if n.Biases[l].Norm() != 0 {
			t.Errorf("layer %d: biases not zeroed", l)
		}
	}
}

func TestInitHeBoundsAndBiases(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 6},
		Layer{Nodes: 4, Activator: ReLU()},
		Layer{Nodes: 2, Activator: Sigmoid()},
	)
	n.InitHe()

	for l := range n.Weights {
		limit := math.Sqrt(6 / float64(n.Layers[l].Nodes))

		if max := n.Weights[l].Map(func(v float64, x, y int) float64 {
			return math.Abs(v)
		}).Max(); max > limit {
			t.Errorf("layer %d: |weight| %g exceeds He limit %g", l, max, limit)
		}

		if n.Biases[l].Norm() != 0 {
			t.Errorf("layer %d: biases not zeroed", l)
		}
	}
}

// TestXavierInitStillLearns makes sure the re-initialized network is a
// working starting point, not just one that satisfies the bounds.
func TestXavierInitStillLearns(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)
	n.InitXavier()

	data := xorData()
	optimizer := Adam(2)
	firstErr := optimizer(n, data)

	var lastErr float64
	for i := 0; i < 500; i++ {
		lastErr = optimizer(n, data)
	}

	if lastErr >= firstErr {
		t.Fatalf("expected training after InitXavier to reduce loss, got firstErr=%f lastErr=%f", firstErr, lastErr)
	}

	if got := n.Accuracy(data); got != 1 {
		t.Errorf("Accuracy after training = %g, want 1", got)
	}
}
