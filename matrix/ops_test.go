package matrix

import (
	"math"
	"testing"
)

func TestIdentity(t *testing.T) {
	m := Identity(3)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			want := 0.0
			if i == j {
				want = 1
			}
			if m.At(i, j) != want {
				t.Fatalf("Identity(3).At(%d, %d) = %g, want %g", i, j, m.At(i, j), want)
			}
		}
	}

	if got := Identity(4).Multiply(NewFromArray([]float64{1, 2, 3, 4})); !got.Equal(NewFromArray([]float64{1, 2, 3, 4})) {
		t.Fatalf("Identity(4) x v changed v: %v", got)
	}

	mustPanic(t, "negative size", func() { Identity(-1) })
}

func TestMeanMinMax(t *testing.T) {
	m := New(2, 2, [][]float64{{4, -2}, {7, 3}})

	if got := m.Mean(); got != 3 {
		t.Errorf("Mean = %g, want 3", got)
	}
	if got := m.Min(); got != -2 {
		t.Errorf("Min = %g, want -2", got)
	}
	if got := m.Max(); got != 7 {
		t.Errorf("Max = %g, want 7", got)
	}

	empty := New(0, 0, nil)
	mustPanic(t, "Mean of empty", func() { empty.Mean() })
	mustPanic(t, "Min of empty", func() { empty.Min() })
	mustPanic(t, "Max of empty", func() { empty.Max() })
}

func TestTrace(t *testing.T) {
	m := New(3, 3, [][]float64{{1, 9, 9}, {9, 2, 9}, {9, 9, 3}})
	if got := m.Trace(); got != 6 {
		t.Errorf("Trace = %g, want 6", got)
	}

	mustPanic(t, "Trace of non-square", func() { New(2, 3, nil).Trace() })
}

func TestNorm(t *testing.T) {
	if got := NewFromArray([]float64{3, 4}).Norm(); math.Abs(got-5) > 1e-12 {
		t.Errorf("Norm([3 4]) = %g, want 5", got)
	}
	if got := New(2, 2, [][]float64{{1, -2}, {2, 4}}).Norm(); math.Abs(got-5) > 1e-12 {
		t.Errorf("Norm = %g, want 5", got)
	}
	if got := New(0, 0, nil).Norm(); got != 0 {
		t.Errorf("Norm of empty = %g, want 0", got)
	}
}

func TestSolve(t *testing.T) {
	// 2x + y = 5, x + 3y = 10 has the unique solution x = 1, y = 3.
	a := New(2, 2, [][]float64{{2, 1}, {1, 3}})
	b := NewFromArray([]float64{5, 10})

	x, ok := a.Solve(b)
	if !ok {
		t.Fatal("Solve reported a solvable system as singular")
	}
	if !x.ApproxEqual(NewFromArray([]float64{1, 3}), 1e-12) {
		t.Fatalf("Solve = %v, want [1 3]", x)
	}

	// Multiple right-hand sides solve in one call, column by column.
	multi, ok := a.Solve(New(2, 2, [][]float64{{5, 2}, {10, 1}}))
	if !ok {
		t.Fatal("Solve reported a solvable multi-RHS system as singular")
	}
	if !a.Multiply(multi).ApproxEqual(New(2, 2, [][]float64{{5, 2}, {10, 1}}), 1e-12) {
		t.Fatalf("A x X != B for multi-RHS solve, got %v", multi)
	}

	if _, ok := New(2, 2, [][]float64{{1, 2}, {2, 4}}).Solve(b); ok {
		t.Error("Solve of singular system reported ok")
	}

	mustPanic(t, "non-square", func() { New(2, 3, nil).Solve(b) })
	mustPanic(t, "row mismatch", func() { a.Solve(NewFromArray([]float64{1, 2, 3})) })
}

func TestInverse(t *testing.T) {
	m := New(3, 3, [][]float64{{2, 0, 1}, {1, 1, 0}, {0, 3, 1}})

	inv, ok := m.Inverse()
	if !ok {
		t.Fatal("Inverse reported an invertible matrix as singular")
	}
	if !m.Multiply(inv).ApproxEqual(Identity(3), 1e-12) {
		t.Fatalf("m x m^-1 != I, got %v", m.Multiply(inv))
	}
	if !inv.Multiply(m).ApproxEqual(Identity(3), 1e-12) {
		t.Fatalf("m^-1 x m != I, got %v", inv.Multiply(m))
	}

	if _, ok := New(2, 2, [][]float64{{1, 2}, {2, 4}}).Inverse(); ok {
		t.Error("Inverse of singular matrix reported ok")
	}

	mustPanic(t, "non-square", func() { New(2, 3, nil).Inverse() })
}

func TestClip(t *testing.T) {
	m := New(2, 2, [][]float64{{-5, 0.5}, {2, 10}})
	got := m.Clip(-1, 1)
	want := New(2, 2, [][]float64{{-1, 0.5}, {1, 1}})
	if !got.Equal(want) {
		t.Errorf("Clip(-1, 1) = %v, want %v", got, want)
	}

	// Clip must not mutate its receiver -- the package contract is that
	// only Randomize, Zero and Set mutate.
	if m.At(0, 0) != -5 {
		t.Error("Clip mutated its receiver")
	}

	mustPanic(t, "low > high", func() { m.Clip(2, 1) })
}
