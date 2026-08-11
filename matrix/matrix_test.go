package matrix

import (
	"math"
	"testing"
)

func mustPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: expected panic, got none", name)
		}
	}()
	f()
}

func TestNew(t *testing.T) {
	m := New(2, 3, [][]float64{{1, 2, 3}, {4, 5, 6}})
	if m.Rows != 2 || m.Columns != 3 {
		t.Fatalf("shape = %dx%d, want 2x3", m.Rows, m.Columns)
	}
	if m.At(0, 0) != 1 || m.At(1, 2) != 6 {
		t.Fatalf("unexpected elements: %v", m)
	}

	z := New(2, 2, nil)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if z.At(i, j) != 0 {
				t.Fatalf("New(2, 2, nil) not zeroed at (%d, %d)", i, j)
			}
		}
	}

	mustPanic(t, "wrong row count", func() { New(3, 2, [][]float64{{1, 2}}) })
	mustPanic(t, "ragged rows", func() { New(2, 2, [][]float64{{1, 2}, {3}}) })
	mustPanic(t, "negative shape", func() { New(-1, 2, nil) })
}

func TestNewCopiesData(t *testing.T) {
	src := [][]float64{{1, 2}, {3, 4}}
	m := New(2, 2, src)
	src[0][0] = 99
	if m.At(0, 0) != 1 {
		t.Fatal("New aliased caller data instead of copying it")
	}
}

func TestNewFromArray(t *testing.T) {
	d := []float64{1, 2, 3}
	m := NewFromArray(d)
	if m.Rows != 3 || m.Columns != 1 {
		t.Fatalf("shape = %dx%d, want 3x1", m.Rows, m.Columns)
	}
	d[0] = 99
	if m.At(0, 0) != 1 {
		t.Fatal("NewFromArray aliased caller data instead of copying it")
	}
}

func TestFlattenUnflattenRoundTrip(t *testing.T) {
	m := New(2, 3, [][]float64{{1, 2, 3}, {4, 5, 6}})
	flat := m.Flatten()
	want := []float64{1, 2, 3, 4, 5, 6}
	for i, v := range want {
		if flat[i] != v {
			t.Fatalf("Flatten()[%d] = %v, want %v", i, flat[i], v)
		}
	}

	back := Unflatten(2, 3, flat)
	if !back.Equal(m) {
		t.Fatalf("Unflatten(Flatten(m)) != m:\n%v\n%v", back, m)
	}

	// Both directions must copy, so neither aliases the other.
	flat[0] = 99
	if m.At(0, 0) != 1 || back.At(0, 0) != 1 {
		t.Fatal("Flatten/Unflatten aliased the slice instead of copying it")
	}

	mustPanic(t, "wrong length", func() { Unflatten(2, 3, []float64{1}) })
}

func TestCopyIsIndependent(t *testing.T) {
	m := New(2, 2, [][]float64{{1, 2}, {3, 4}})
	c := m.Copy()
	c.Set(0, 0, 99)
	if m.At(0, 0) != 1 {
		t.Fatal("mutating a copy changed the original")
	}
}

func TestAtSetBounds(t *testing.T) {
	m := Zeros(2, 3)
	m.Set(1, 2, 7)
	if m.At(1, 2) != 7 {
		t.Fatalf("At(1, 2) = %v after Set, want 7", m.At(1, 2))
	}

	// (0, 3) is out of range even though its flat offset (3) is valid, so
	// unchecked index math would silently read row 1.
	mustPanic(t, "column overflow", func() { m.At(0, 3) })
	mustPanic(t, "row overflow", func() { m.At(2, 0) })
	mustPanic(t, "negative index", func() { m.At(-1, 0) })
	mustPanic(t, "set out of range", func() { m.Set(0, 3, 1) })
}

func TestMapPassesCoordinates(t *testing.T) {
	m := Zeros(2, 3)
	got := m.Map(func(val float64, x, y int) float64 {
		return float64(x*10 + y)
	})
	want := New(2, 3, [][]float64{{0, 1, 2}, {10, 11, 12}})
	if !got.Equal(want) {
		t.Fatalf("Map coordinates wrong:\n%v\nwant:\n%v", got, want)
	}
}

func TestFold(t *testing.T) {
	m := New(2, 2, [][]float64{{1, 2}, {3, 4}})
	got := m.Fold(func(acc, val float64, x, y int) float64 { return acc + val }, 10)
	if got != 20 {
		t.Fatalf("Fold sum = %v, want 20", got)
	}
}

func TestElementwiseOps(t *testing.T) {
	a := New(2, 2, [][]float64{{1, 2}, {3, 4}})
	b := New(2, 2, [][]float64{{10, 20}, {30, 40}})

	if got, want := a.AddMatrix(b), New(2, 2, [][]float64{{11, 22}, {33, 44}}); !got.Equal(want) {
		t.Fatalf("AddMatrix:\n%v\nwant:\n%v", got, want)
	}
	if got, want := b.SubtractMatrix(a), New(2, 2, [][]float64{{9, 18}, {27, 36}}); !got.Equal(want) {
		t.Fatalf("SubtractMatrix:\n%v\nwant:\n%v", got, want)
	}
	if got, want := a.HadamardProduct(b), New(2, 2, [][]float64{{10, 40}, {90, 160}}); !got.Equal(want) {
		t.Fatalf("HadamardProduct:\n%v\nwant:\n%v", got, want)
	}
	if got, want := a.Add(1), New(2, 2, [][]float64{{2, 3}, {4, 5}}); !got.Equal(want) {
		t.Fatalf("Add:\n%v\nwant:\n%v", got, want)
	}
	if got, want := a.Subtract(1), New(2, 2, [][]float64{{0, 1}, {2, 3}}); !got.Equal(want) {
		t.Fatalf("Subtract:\n%v\nwant:\n%v", got, want)
	}
	if got, want := a.Scale(2), New(2, 2, [][]float64{{2, 4}, {6, 8}}); !got.Equal(want) {
		t.Fatalf("Scale:\n%v\nwant:\n%v", got, want)
	}
	if got, want := a.Divide(2), New(2, 2, [][]float64{{0.5, 1}, {1.5, 2}}); !got.Equal(want) {
		t.Fatalf("Divide:\n%v\nwant:\n%v", got, want)
	}
	if got := a.Sum(); got != 10 {
		t.Fatalf("Sum = %v, want 10", got)
	}

	c := Zeros(2, 3)
	mustPanic(t, "AddMatrix shape", func() { a.AddMatrix(c) })
	mustPanic(t, "SubtractMatrix shape", func() { a.SubtractMatrix(c) })
	mustPanic(t, "HadamardProduct shape", func() { a.HadamardProduct(c) })
}

func TestTranspose(t *testing.T) {
	m := New(2, 3, [][]float64{{1, 2, 3}, {4, 5, 6}})
	got := m.Transpose()
	want := New(3, 2, [][]float64{{1, 4}, {2, 5}, {3, 6}})
	if !got.Equal(want) {
		t.Fatalf("Transpose:\n%v\nwant:\n%v", got, want)
	}
	if !m.Transpose().Transpose().Equal(m) {
		t.Fatal("double transpose is not the identity")
	}
}

func TestMultiply(t *testing.T) {
	a := New(2, 3, [][]float64{{1, 2, 3}, {4, 5, 6}})
	b := New(3, 2, [][]float64{{7, 8}, {9, 10}, {11, 12}})
	got := a.Multiply(b)
	want := New(2, 2, [][]float64{{58, 64}, {139, 154}})
	if !got.Equal(want) {
		t.Fatalf("Multiply:\n%v\nwant:\n%v", got, want)
	}

	// The column-vector fast path must agree with the general kernel.
	v := NewFromArray([]float64{1, 2, 3})
	gotVec := a.Multiply(v)
	wantVec := NewFromArray([]float64{14, 32})
	if !gotVec.Equal(wantVec) {
		t.Fatalf("Multiply vector:\n%v\nwant:\n%v", gotVec, wantVec)
	}

	// Outer product: column vector times row vector.
	col := NewFromArray([]float64{1, 2})
	row := New(1, 3, [][]float64{{3, 4, 5}})
	gotOuter := col.Multiply(row)
	wantOuter := New(2, 3, [][]float64{{3, 4, 5}, {6, 8, 10}})
	if !gotOuter.Equal(wantOuter) {
		t.Fatalf("outer product:\n%v\nwant:\n%v", gotOuter, wantOuter)
	}

	id := New(3, 3, [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})
	if !a.Multiply(id).Equal(a) {
		t.Fatal("multiplying by identity changed the matrix")
	}

	mustPanic(t, "Multiply shape", func() { a.Multiply(a) })
}

func TestDet(t *testing.T) {
	if got := New(1, 1, [][]float64{{5}}).Det(); got != 5 {
		t.Fatalf("Det 1x1 = %v, want 5", got)
	}
	if got := New(2, 2, [][]float64{{1, 2}, {3, 4}}).Det(); math.Abs(got-(-2)) > 1e-12 {
		t.Fatalf("Det 2x2 = %v, want -2", got)
	}
	m := New(3, 3, [][]float64{{6, 1, 1}, {4, -2, 5}, {2, 8, 7}})
	if got := m.Det(); math.Abs(got-(-306)) > 1e-9 {
		t.Fatalf("Det 3x3 = %v, want -306", got)
	}
	singular := New(2, 2, [][]float64{{1, 2}, {2, 4}})
	if got := singular.Det(); got != 0 {
		t.Fatalf("Det of singular matrix = %v, want 0", got)
	}
	// A leading zero forces the pivoting branch.
	pivoted := New(2, 2, [][]float64{{0, 1}, {1, 0}})
	if got := pivoted.Det(); math.Abs(got-(-1)) > 1e-12 {
		t.Fatalf("Det with pivoting = %v, want -1", got)
	}

	mustPanic(t, "Det non-square", func() { Zeros(2, 3).Det() })
}

func TestRandomizeRange(t *testing.T) {
	m := Zeros(100, 100)
	m.Randomize(-1, 1)

	min, max := math.Inf(1), math.Inf(-1)
	for _, v := range m.Flatten() {
		if v < -1 || v >= 1 {
			t.Fatalf("Randomize(-1, 1) produced %v, outside [-1, 1)", v)
		}
		min = math.Min(min, v)
		max = math.Max(max, v)
	}

	// With 10k uniform draws, both halves of the range are hit with
	// overwhelming probability; this is what catches the classic
	// rand.Float64()*max + min bug, which never exceeds min+max.
	if min > -0.5 || max < 0.5 {
		t.Fatalf("Randomize(-1, 1) looks skewed: observed range [%v, %v]", min, max)
	}
}

func TestZero(t *testing.T) {
	m := New(2, 2, [][]float64{{1, 2}, {3, 4}})
	m.Zero()
	if !m.Equal(Zeros(2, 2)) {
		t.Fatalf("Zero left values behind:\n%v", m)
	}
}

func TestEqualAndApproxEqual(t *testing.T) {
	a := New(2, 2, [][]float64{{1, 2}, {3, 4}})
	if !a.Equal(a.Copy()) {
		t.Fatal("copy not Equal to original")
	}
	if a.Equal(Zeros(2, 3)) {
		t.Fatal("Equal ignored shape")
	}
	if a.Equal(Zeros(2, 2)) {
		t.Fatal("Equal ignored values")
	}

	b := a.Add(1e-9)
	if a.Equal(b) {
		t.Fatal("Equal should be exact")
	}
	if !a.ApproxEqual(b, 1e-8) {
		t.Fatal("ApproxEqual rejected values within tolerance")
	}
	if a.ApproxEqual(b, 1e-10) {
		t.Fatal("ApproxEqual accepted values outside tolerance")
	}
}

func TestZeroValueMatrix(t *testing.T) {
	var m Matrix
	if got := m.Sum(); got != 0 {
		t.Fatalf("zero value Sum = %v, want 0", got)
	}
	if len(m.Flatten()) != 0 {
		t.Fatal("zero value Flatten should be empty")
	}
	if !m.Transpose().Equal(m) {
		t.Fatal("zero value Transpose should stay empty")
	}
}
