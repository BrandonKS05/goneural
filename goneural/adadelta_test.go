package goneural

import (
	"math"
	"math/rand"
	"testing"
)

func TestAdaDeltaReducesLoss(t *testing.T) {
	rand.Seed(7)
	// AdaDelta bootstraps its step size from nothing, so it needs a longer
	// runway than the fixed-rate optimizers before XOR separates.
	testOptimizerLearnsXOR(t, AdaDelta(2), 6000)
}

// TestAdaDeltaStepsGrowFromRest pins the bootstrap behaviour: from zeroed
// accumulators the first step is on the order of sqrt(epsilon), far smaller
// than the raw gradient, and repeated identical gradients grow the step as
// update history accumulates.
func TestAdaDeltaStepsGrowFromRest(t *testing.T) {
	gradSquares := make([]float64, 1)
	stepSquares := make([]float64, 1)

	first := adaDeltaStep([]float64{0.5}, gradSquares, stepSquares, 1, 0.95, 1e-6)[0]
	if math.Abs(first) >= 0.01 {
		t.Fatalf("first step from rest = %g, expected a tiny bootstrap step", first)
	}

	prev := first
	for i := 0; i < 10; i++ {
		next := adaDeltaStep([]float64{0.5}, gradSquares, stepSquares, 1, 0.95, 1e-6)[0]
		if next <= prev {
			t.Fatalf("iteration %d: expected growing steps while history builds, got %g then %g", i, prev, next)
		}
		prev = next
	}
}
