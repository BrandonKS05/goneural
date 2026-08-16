package goneural

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

func TestLAMBReducesLoss(t *testing.T) {
	rand.Seed(17)
	testOptimizerLearnsXOR(t, LAMB(2, 0.05, 0), 500)
}

// TestLAMBStepIsAFixedFractionOfLayerScale is LAMB's defining invariant. After
// the trust ratio is applied, the size of a layer's update is lr * ||w|| --
// it depends on the layer's own weight scale and not at all on how large the
// raw Adam direction happened to be. That is what makes one global learning
// rate mean the same thing in every layer.
func TestLAMBStepIsAFixedFractionOfLayerScale(t *testing.T) {
	const (
		lr    = 0.01
		beta1 = 0.9
		beta2 = 0.999
		eps   = 1e-6
	)

	params := []float64{3, -4, 12, 0} // norm 13
	paramNorm := matrix.Unflatten(2, 2, params).Norm()

	for _, gradScale := range []float64{1e-3, 1, 1e3} {
		m := make([]float64, len(params))
		v := make([]float64, len(params))
		grads := []float64{gradScale, -gradScale, gradScale, gradScale}

		update := lambDirection(grads, m, v, params, 1, 0, beta1, beta2, eps, 1, 1)
		trust := lambTrustRatio(paramNorm, matrix.Unflatten(2, 2, update).Norm())

		for k := range update {
			update[k] *= lr * trust
		}

		got := matrix.Unflatten(2, 2, update).Norm()
		want := lr * paramNorm

		if math.Abs(got-want) > 1e-9 {
			t.Errorf("gradient scale %g: applied step norm = %g, want %g", gradScale, got, want)
		}
	}
}

// TestLAMBTrustRatioHandlesDegenerateNorms covers the paper's convention that
// a vanishing weight norm or update norm bypasses the ratio instead of
// producing a zero or infinite step.
func TestLAMBTrustRatioHandlesDegenerateNorms(t *testing.T) {
	if got := lambTrustRatio(0, 5); got != 1 {
		t.Errorf("zero weight norm gave trust %g, want 1", got)
	}
	if got := lambTrustRatio(5, 0); got != 1 {
		t.Errorf("zero update norm gave trust %g, want 1", got)
	}
	if got := lambTrustRatio(10, 4); got != 2.5 {
		t.Errorf("trust ratio = %g, want 2.5", got)
	}
}

// TestLAMBWeightDecayEntersBeforeTheRatio checks the decay is part of the
// direction whose norm forms the trust ratio, not a separate step bolted on
// afterwards -- otherwise the ratio would be computed for an update that is
// not the one actually applied.
func TestLAMBWeightDecayEntersBeforeTheRatio(t *testing.T) {
	const (
		beta1 = 0.9
		beta2 = 0.999
		eps   = 1e-6
	)

	params := []float64{2}
	grads := []float64{0} // no gradient signal at all

	m := make([]float64, 1)
	v := make([]float64, 1)
	update := lambDirection(grads, m, v, params, 1, 0.5, beta1, beta2, eps, 1, 1)

	// With a zero gradient the entire direction is decay: 0.5 * 2 = 1.
	if math.Abs(update[0]-1) > 1e-12 {
		t.Fatalf("decay-only direction = %g, want 1", update[0])
	}
}
