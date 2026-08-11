package goneural

import (
	"math/rand"
	"testing"
)

func TestAdaGradReducesLoss(t *testing.T) {
	rand.Seed(7)
	testOptimizerLearnsXOR(t, AdaGrad(2, 0.5), 2000)
}

// TestAdaGradShrinksEffectiveStep pins AdaGrad's defining property: since
// the squared-gradient accumulator only ever grows, feeding the same
// gradient forever must produce strictly smaller steps each time.
func TestAdaGradShrinksEffectiveStep(t *testing.T) {
	sumSquares := make([]float64, 1)
	grad := []float64{0.5}

	prev := adaGradStep(grad, sumSquares, 1, 0.1, 1e-8)[0]
	for i := 0; i < 5; i++ {
		next := adaGradStep(grad, sumSquares, 1, 0.1, 1e-8)[0]
		if next >= prev {
			t.Fatalf("step %d: expected shrinking steps, got %g then %g", i, prev, next)
		}
		prev = next
	}
}
