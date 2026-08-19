package matrix

import (
	"math"
	"math/rand"
	"testing"
)

// TestSymmetricEigenDiagonal is the trivial case: a diagonal matrix is
// already decomposed, so the eigenvalues are its diagonal (sorted) and the
// eigenvectors are the axes.
func TestSymmetricEigenDiagonal(t *testing.T) {
	m := New(3, 3, [][]float64{
		{2, 0, 0},
		{0, 5, 0},
		{0, 0, 1},
	})

	values, vectors, ok := m.SymmetricEigen()
	if !ok {
		t.Fatal("SymmetricEigen did not converge on a diagonal matrix")
	}

	for i, want := range []float64{5, 2, 1} {
		if math.Abs(values[i]-want) > 1e-12 {
			t.Errorf("values[%d] = %g, want %g", i, values[i], want)
		}
	}

	// The leading eigenvector must be the axis carrying the 5.
	for row, want := range []float64{0, 1, 0} {
		if math.Abs(math.Abs(vectors.At(row, 0))-want) > 1e-12 {
			t.Errorf("leading eigenvector[%d] = %g, want +-%g", row, vectors.At(row, 0), want)
		}
	}
}

// TestSymmetricEigenKnown2x2 checks a case with an exact hand answer:
// [[2, 1], [1, 2]] has eigenvalues 3 and 1 along (1, 1) and (1, -1).
func TestSymmetricEigenKnown2x2(t *testing.T) {
	m := New(2, 2, [][]float64{
		{2, 1},
		{1, 2},
	})

	values, vectors, ok := m.SymmetricEigen()
	if !ok {
		t.Fatal("SymmetricEigen did not converge")
	}

	for i, want := range []float64{3, 1} {
		if math.Abs(values[i]-want) > 1e-12 {
			t.Errorf("values[%d] = %g, want %g", i, values[i], want)
		}
	}

	root := 1 / math.Sqrt(2)
	if math.Abs(math.Abs(vectors.At(0, 0))-root) > 1e-12 || math.Abs(math.Abs(vectors.At(1, 0))-root) > 1e-12 {
		t.Errorf("leading eigenvector = (%g, %g), want +-(%g, %g)",
			vectors.At(0, 0), vectors.At(1, 0), root, root)
	}
	// The two axes are the sum and difference directions, so their entries
	// must have matching signs in one and opposing signs in the other.
	if vectors.At(0, 0)*vectors.At(1, 0) < 0 == (vectors.At(0, 1)*vectors.At(1, 1) < 0) {
		t.Error("eigenvectors are not the sum and difference directions")
	}
}

// randomSymmetric builds a symmetric matrix from a random one, which is
// the standard way to get a valid test input: A + A^T is symmetric for any
// A.
func randomSymmetric(n int) Matrix {
	m := zeros(n, n)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := rand.NormFloat64()
			m.Set(i, j, v)
			m.Set(j, i, v)
		}
	}
	return m
}

// TestSymmetricEigenReconstructs is the property that actually matters:
// V * diag(values) * V^T must give the original matrix back, and V must be
// orthonormal.
func TestSymmetricEigenReconstructs(t *testing.T) {
	rand.Seed(7)

	for _, n := range []int{1, 2, 4, 6} {
		m := randomSymmetric(n)

		values, vectors, ok := m.SymmetricEigen()
		if !ok {
			t.Fatalf("n=%d: SymmetricEigen did not converge", n)
		}

		diagonal := zeros(n, n)
		for i, v := range values {
			diagonal.Set(i, i, v)
		}

		if got := vectors.Multiply(diagonal).Multiply(vectors.Transpose()); !got.ApproxEqual(m, 1e-9) {
			t.Errorf("n=%d: reconstruction differs from the original:\ngot %v\nwant %v", n, got, m)
		}

		if got := vectors.Transpose().Multiply(vectors); !got.ApproxEqual(Identity(n), 1e-9) {
			t.Errorf("n=%d: eigenvectors are not orthonormal: V^T V = %v", n, got)
		}

		// Descending order is part of the contract.
		for i := 1; i < len(values); i++ {
			if values[i] > values[i-1] {
				t.Errorf("n=%d: values[%d] = %g exceeds the preceding %g", n, i, values[i], values[i-1])
			}
		}
	}
}

// TestSymmetricEigenMatchesTraceAndDeterminant cross-checks the spectrum
// against two invariants computed by unrelated code paths.
func TestSymmetricEigenMatchesTraceAndDeterminant(t *testing.T) {
	rand.Seed(11)

	m := randomSymmetric(4)
	values, _, ok := m.SymmetricEigen()
	if !ok {
		t.Fatal("SymmetricEigen did not converge")
	}

	sum, product := 0.0, 1.0
	for _, v := range values {
		sum += v
		product *= v
	}

	if math.Abs(sum-m.Trace()) > 1e-9 {
		t.Errorf("eigenvalues sum to %g, want the trace %g", sum, m.Trace())
	}
	if math.Abs(product-m.Det()) > 1e-9 {
		t.Errorf("eigenvalues multiply to %g, want the determinant %g", product, m.Det())
	}
}

// TestSymmetricEigenRepeatedValues covers the degenerate spectrum, where
// the eigenvectors are not unique and a rotation can find no angle to
// prefer: the identity matrix.
func TestSymmetricEigenRepeatedValues(t *testing.T) {
	m := Identity(3).Scale(4)

	values, vectors, ok := m.SymmetricEigen()
	if !ok {
		t.Fatal("SymmetricEigen did not converge on a scaled identity")
	}

	for i, v := range values {
		if math.Abs(v-4) > 1e-12 {
			t.Errorf("values[%d] = %g, want 4", i, v)
		}
	}
	if got := vectors.Transpose().Multiply(vectors); !got.ApproxEqual(Identity(3), 1e-12) {
		t.Errorf("eigenvectors are not orthonormal: %v", got)
	}
}

func TestSymmetricEigenRejectsBadInput(t *testing.T) {
	for name, f := range map[string]func(){
		"non-square":  func() { New(2, 3, nil).SymmetricEigen() },
		"empty":       func() { New(0, 0, nil).SymmetricEigen() },
		"unsymmetric": func() { New(2, 2, [][]float64{{1, 2}, {3, 4}}).SymmetricEigen() },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: expected a panic, got none", name)
				}
			}()
			f()
		}()
	}
}
