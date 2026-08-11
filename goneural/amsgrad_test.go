package goneural

import (
	"math/rand"
	"testing"
)

func TestAMSGradReducesLoss(t *testing.T) {
	rand.Seed(7)
	testOptimizerLearnsXOR(t, AMSGrad(2, 0.05), 500)
}

// TestAMSGradKeepsSecondMomentHighWaterMark pins the defining difference
// from Adam: after one large gradient, a stream of tiny gradients decays v
// but must not decay the vMax the step divides by.
func TestAMSGradKeepsSecondMomentHighWaterMark(t *testing.T) {
	const (
		lr      = 0.1
		beta1   = 0.9
		beta2   = 0.999
		epsilon = 1e-8
	)

	m := make([]float64, 1)
	v := make([]float64, 1)
	vMax := make([]float64, 1)

	amsGradStep([]float64{10}, m, v, vMax, 1, lr, beta1, beta2, epsilon, 1, 1)
	peak := vMax[0]

	for i := 0; i < 20; i++ {
		amsGradStep([]float64{1e-4}, m, v, vMax, 1, lr, beta1, beta2, epsilon, 1, 1)
	}

	if v[0] >= peak {
		t.Fatalf("expected v to decay below its peak %g, got %g", peak, v[0])
	}
	if vMax[0] < peak {
		t.Fatalf("high-water mark decayed: vMax %g fell below peak %g", vMax[0], peak)
	}
}
