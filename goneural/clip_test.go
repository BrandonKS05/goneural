package goneural

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

func TestClipByGlobalNorm(t *testing.T) {
	// Two matrices with norms 3 and 4 have a global norm of 5.
	ms := []matrix.Matrix{
		matrix.NewFromArray([]float64{3}),
		matrix.NewFromArray([]float64{4}),
	}

	// Above the threshold: everything shrinks by the same factor, so the
	// global norm lands exactly on maxNorm and ratios are preserved.
	clipped := ClipByGlobalNorm(ms, 1)
	got := math.Sqrt(clipped[0].Norm()*clipped[0].Norm() + clipped[1].Norm()*clipped[1].Norm())
	if math.Abs(got-1) > 1e-12 {
		t.Errorf("global norm after clipping = %g, want 1", got)
	}
	if ratio := clipped[1].At(0, 0) / clipped[0].At(0, 0); math.Abs(ratio-4.0/3.0) > 1e-12 {
		t.Errorf("clipping bent the direction: ratio = %g, want 4/3", ratio)
	}

	// At or below the threshold the input passes through untouched.
	untouched := ClipByGlobalNorm(ms, 10)
	if untouched[0].At(0, 0) != 3 || untouched[1].At(0, 0) != 4 {
		t.Errorf("clipping altered gradients already under the limit: %v", untouched)
	}

	// maxNorm <= 0 disables clipping.
	disabled := ClipByGlobalNorm(ms, 0)
	if disabled[0].At(0, 0) != 3 {
		t.Error("maxNorm 0 should disable clipping")
	}
}

func TestMomentumWithClippingStillLearns(t *testing.T) {
	rand.Seed(7)

	o := NewMomentumOptimizer(2, 0.5, 0.9)
	o.MaxGradNorm = 1

	testOptimizerLearnsXOR(t, o.Optimize, 800)
}

// TestMomentumClippingBoundsTheStep pins the mechanism: with a tiny
// MaxGradNorm every velocity injection is at most that norm, so after one
// epoch from rest the first step can't exceed lr * maxNorm per batch.
func TestMomentumClippingBoundsTheStep(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	const maxNorm = 1e-3

	o := NewMomentumOptimizer(4, 0.5, 0.9)
	o.MaxGradNorm = maxNorm

	before := make([]matrix.Matrix, len(n.Weights))
	for l := range n.Weights {
		before[l] = n.Weights[l].Copy()
	}

	o.Optimize(n, xorData()) // one epoch = one batch of all four samples

	sumSquares := 0.0
	for l := range n.Weights {
		d := n.Weights[l].SubtractMatrix(before[l]).Norm()
		sumSquares += d * d
	}

	if step := math.Sqrt(sumSquares); step > o.LearningRate*maxNorm+1e-12 {
		t.Fatalf("first step norm %g exceeds lr*maxNorm = %g", step, o.LearningRate*maxNorm)
	}
}
