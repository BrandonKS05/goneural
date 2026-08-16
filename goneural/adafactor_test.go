package goneural

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

func TestAdafactorReducesLoss(t *testing.T) {
	rand.Seed(19)
	testOptimizerLearnsXOR(t, Adafactor(2, 0.05), 500)
}

// TestAdafactorReconstructsRankOneSecondMomentExactly is the mathematical
// heart of the method. When the squared gradient really is rank one --
// g[i][j] = a[i]*b[j], so g^2[i][j] = a^2[i]*b^2[j] -- the row/column
// factorization is not an approximation at all, and the reconstruction
// R[i]*C[j]/total recovers g^2 exactly. The update is then g/sqrt(g^2), which
// is elementwise sign, so a wholly positive gradient must yield an update of
// exactly ones.
func TestAdafactorReconstructsRankOneSecondMomentExactly(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5}

	grad := matrix.New(len(a), len(b), nil)
	for i := range a {
		for j := range b {
			grad.Set(i, j, a[i]*b[j])
		}
	}

	rowAcc := make([]float64, len(a))
	colAcc := make([]float64, len(b))

	// beta2t = 0 at the first step, epsilon 0 so the factorization is exact.
	update := adafactorFactoredUpdate(grad, rowAcc, colAcc, 1, 0, 0, 1)

	for k, got := range update {
		if math.Abs(got-1) > 1e-9 {
			t.Errorf("update[%d] = %g, want 1 (exact rank-one reconstruction)", k, got)
		}
	}
}

// TestAdafactorStateIsLinearInLayerSize is the reason the optimizer exists:
// a rows x cols weight matrix must cost rows + cols numbers of state, not
// rows * cols.
func TestAdafactorStateIsLinearInLayerSize(t *testing.T) {
	rand.Seed(23)

	n := New(0.1, MSE(),
		Layer{Nodes: 8},
		Layer{Nodes: 16, Activator: Sigmoid()},
		Layer{Nodes: 4, Activator: Sigmoid()},
	)

	o := NewAdafactorOptimizer(2, 0.01)
	o.Optimize(n, DataSet{{Inputs: make([]float64, 8), Targets: make([]float64, 4)}})

	stored := 0
	for l := range o.rowW {
		stored += len(o.rowW[l]) + len(o.colW[l]) + len(o.vB[l])
	}

	full := 0
	for l := range n.Weights {
		full += n.Weights[l].Rows*n.Weights[l].Columns + n.Biases[l].Rows
	}

	// 16+8 + 4+16 for the two weight matrices, plus 16+4 of bias state.
	const want = 64
	if stored != want {
		t.Fatalf("Adafactor stored %d floats of second-moment state, want %d", stored, want)
	}

	if stored >= full {
		t.Fatalf("Adafactor stored %d floats, no better than the full %d", stored, full)
	}
}

// TestAdafactorClipsByRMSNotByNorm distinguishes the clipping rule from the
// global-norm clipping already in the package. Two updates with the same
// typical element but different lengths must be treated identically; a
// norm-based rule would punish the longer one.
func TestAdafactorClipsByRMSNotByNorm(t *testing.T) {
	short := adafactorClip([]float64{4, 4}, 1)
	long := adafactorClip([]float64{4, 4, 4, 4, 4, 4, 4, 4}, 1)

	if math.Abs(short[0]-1) > 1e-12 {
		t.Errorf("short update clipped to %g, want 1", short[0])
	}
	if math.Abs(long[0]-1) > 1e-12 {
		t.Errorf("long update clipped to %g, want 1", long[0])
	}

	// An update already under the threshold passes through untouched.
	small := adafactorClip([]float64{0.25, -0.25}, 1)
	if small[0] != 0.25 || small[1] != -0.25 {
		t.Errorf("under-threshold update was modified: %v", small)
	}
}

// TestAdafactorDecayStartsAtZero pins the reason there is no bias correction
// anywhere in this optimizer: at t=1 the accumulator is the raw squared
// gradient, so there is no zero-initialization to correct for.
func TestAdafactorDecayStartsAtZero(t *testing.T) {
	if got := adafactorDecay(1, 0.8); got != 0 {
		t.Fatalf("decay at t=1 = %g, want 0", got)
	}

	previous := 0.0
	for step := 1; step <= 1000; step++ {
		got := adafactorDecay(step, 0.8)
		if got < previous {
			t.Fatalf("decay decreased at step %d: %g after %g", step, got, previous)
		}
		if got >= 1 {
			t.Fatalf("decay reached %g at step %d, must stay below 1", got, step)
		}
		previous = got
	}
}
