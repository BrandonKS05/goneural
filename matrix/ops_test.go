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
