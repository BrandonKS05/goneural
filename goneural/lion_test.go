package goneural

import (
	"math"
	"math/rand"
	"testing"
)

func TestLionReducesLoss(t *testing.T) {
	rand.Seed(7)
	// Lion moves every parameter by the full learning rate each update, so
	// it wants a much smaller rate than the Adam family.
	testOptimizerLearnsXOR(t, Lion(2, 0.01), 800)
}

// TestLionStepsAreUniform pins the sign-based property: whatever the
// gradient magnitudes, every emitted step is exactly +-lr (or 0 when the
// interpolation cancels to nothing).
func TestLionStepsAreUniform(t *testing.T) {
	const lr = 0.01

	m := make([]float64, 4)
	steps := lionStep([]float64{100, -0.001, 3.7, 0}, m, 1, lr, 0.9, 0.99)

	wants := []float64{lr, -lr, lr, 0}
	for k, want := range wants {
		if steps[k] != want {
			t.Errorf("step[%d] = %g, want %g", k, steps[k], want)
		}
	}
}

// TestLionMomentumCanOverruleGradient checks the momentum's role: after
// enough history in one direction, a single small opposing gradient must
// not flip the sign of the step, while a large one must.
func TestLionMomentumCanOverruleGradient(t *testing.T) {
	const lr = 0.01

	m := make([]float64, 1)
	for i := 0; i < 50; i++ {
		lionStep([]float64{1}, m, 1, lr, 0.9, 0.99)
	}

	if m[0] <= 0 {
		t.Fatalf("momentum after positive gradients = %g, want positive", m[0])
	}

	// beta1*m + (1-beta1)*g with a small negative g stays positive.
	small := lionStep([]float64{-0.1}, []float64{m[0]}, 1, lr, 0.9, 0.99)[0]
	if small != lr {
		t.Errorf("small opposing gradient flipped the step: %g", small)
	}

	// A gradient large enough to swamp the momentum flips it.
	large := lionStep([]float64{-100}, []float64{m[0]}, 1, lr, 0.9, 0.99)[0]
	if large != -lr {
		t.Errorf("large opposing gradient failed to flip the step: %g", large)
	}

	if math.Abs(small) != math.Abs(large) {
		t.Errorf("step magnitudes differ: %g vs %g", small, large)
	}
}
