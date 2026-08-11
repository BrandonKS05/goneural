package goneural

import (
	"math"
	"math/rand"
	"testing"
)

func TestAdamaxReducesLoss(t *testing.T) {
	rand.Seed(7)
	testOptimizerLearnsXOR(t, Adamax(2, 0.05), 500)
}

// TestAdamaxInfinityNormDecaysGeometrically pins the difference from both
// Adam and AMSGrad: after one large gradient, a stream of tiny ones leaves
// the denominator at big * beta2^k -- decaying, unlike AMSGrad's permanent
// high-water mark, but never averaged down the way Adam's second moment is.
func TestAdamaxInfinityNormDecaysGeometrically(t *testing.T) {
	const (
		lr    = 0.01
		beta1 = 0.9
		beta2 = 0.999
	)

	m := make([]float64, 1)
	u := make([]float64, 1)

	adamaxStep([]float64{10}, m, u, 1, lr, beta1, beta2, 0, 1)
	if u[0] != 10 {
		t.Fatalf("u after |g|=10 is %g, want 10", u[0])
	}

	const rounds = 5
	for i := 0; i < rounds; i++ {
		adamaxStep([]float64{1e-9}, m, u, 1, lr, beta1, beta2, 0, 1)
	}

	want := 10 * math.Pow(beta2, rounds)
	if math.Abs(u[0]-want) > 1e-9 {
		t.Fatalf("u after %d tiny gradients = %g, want %g", rounds, u[0], want)
	}

	// A fresh large gradient must take over the norm immediately.
	adamaxStep([]float64{25}, m, u, 1, lr, beta1, beta2, 0, 1)
	if u[0] != 25 {
		t.Fatalf("u after |g|=25 is %g, want 25", u[0])
	}
}
