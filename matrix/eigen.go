package matrix

import (
	"math"
	"sort"
)

// jacobiSweeps bounds the number of passes over the off-diagonal entries.
// Cyclic Jacobi converges quadratically once the off-diagonal mass is
// small, and 50 sweeps is far past what any well-formed symmetric matrix
// needs -- it exists only so a pathological input cannot spin forever.
const jacobiSweeps = 50

// SymmetricEigen computes the full eigendecomposition of a symmetric
// matrix by the cyclic Jacobi method, returning the eigenvalues in
// descending order and a matrix whose *columns* are the matching unit
// eigenvectors, so that m = V * diag(values) * V^T. The bool reports
// whether the iteration converged; a false means the returned values are
// the best available approximation rather than a solved decomposition.
//
// Jacobi works by repeatedly picking the largest off-diagonal entry and
// applying the plane rotation that zeroes it, chipping away at the
// off-diagonal mass until only the diagonal -- the eigenvalues -- is left,
// with the accumulated rotations forming the eigenvectors. It is slower
// than the algorithms a serious linear-algebra library would use, but it
// is short, needs no external dependency, and is unusually accurate on the
// small symmetric matrices this package deals in (covariance matrices,
// above all).
//
// It panics unless the matrix is square and symmetric, since an
// unsymmetric matrix can have complex eigenvalues that this signature
// cannot express.
func (m Matrix) SymmetricEigen() ([]float64, Matrix, bool) {
	if m.Rows != m.Columns {
		panic("matrix: SymmetricEigen of a non-square matrix")
	}
	if m.Rows == 0 {
		panic("matrix: SymmetricEigen of an empty matrix")
	}

	const symmetryTolerance = 1e-9
	for i := 0; i < m.Rows; i++ {
		for j := i + 1; j < m.Columns; j++ {
			if math.Abs(m.At(i, j)-m.At(j, i)) > symmetryTolerance {
				panic("matrix: SymmetricEigen of an unsymmetric matrix")
			}
		}
	}

	n := m.Rows
	a := m.Copy()    // rotated toward diagonal in place
	v := Identity(n) // accumulates the rotations
	converged := false

	for sweep := 0; sweep < jacobiSweeps && !converged; sweep++ {
		// The convergence measure is the off-diagonal mass, which every
		// rotation strictly reduces (rotations are orthogonal, so the total
		// Frobenius norm is invariant -- whatever leaves the off-diagonal
		// has to arrive on the diagonal).
		off := 0.0
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				off += 2 * a.At(i, j) * a.At(i, j)
			}
		}
		if off <= 1e-30 {
			converged = true
			break
		}

		for p := 0; p < n-1; p++ {
			for q := p + 1; q < n; q++ {
				apq := a.At(p, q)
				if apq == 0 {
					continue
				}

				// The rotation angle that annihilates a[p][q], written in
				// the numerically stable form: t is the tangent of the
				// rotation, chosen as the smaller root so that the rotation
				// stays below 45 degrees and no cancellation occurs.
				theta := (a.At(q, q) - a.At(p, p)) / (2 * apq)
				t := 1.0
				if theta != 0 {
					t = math.Copysign(1, theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				}

				c := 1 / math.Sqrt(t*t+1)
				s := t * c

				jacobiRotate(&a, &v, p, q, c, s)
			}
		}
	}

	// Read the eigenvalues off the (now nearly) diagonal matrix and order
	// them largest first, which is what every consumer of a covariance
	// decomposition wants: the leading directions of variance.
	order := make([]int, n)
	values := make([]float64, n)
	for i := range order {
		order[i] = i
		values[i] = a.At(i, i)
	}
	sort.SliceStable(order, func(x, y int) bool { return values[order[x]] > values[order[y]] })

	sorted := make([]float64, n)
	vectors := zeros(n, n)
	for rank, index := range order {
		sorted[rank] = values[index]
		for row := 0; row < n; row++ {
			vectors.Set(row, rank, v.At(row, index))
		}
	}

	return sorted, vectors, converged
}

// jacobiRotate applies one Jacobi rotation of angle (c, s) in the (p, q)
// plane to the working matrix a, and accumulates it into the eigenvector
// matrix v. Only the entries the rotation can touch are updated: rows and
// columns p and q.
func jacobiRotate(a, v *Matrix, p, q int, c, s float64) {
	n := a.Rows

	app, aqq, apq := a.At(p, p), a.At(q, q), a.At(p, q)

	// The 2x2 block, in closed form. a[p][q] is set to exactly zero rather
	// than to whatever rounding leaves behind, which is the whole point of
	// the rotation and keeps the convergence measure honest.
	a.Set(p, p, c*c*app-2*s*c*apq+s*s*aqq)
	a.Set(q, q, s*s*app+2*s*c*apq+c*c*aqq)
	a.Set(p, q, 0)
	a.Set(q, p, 0)

	// The rest of rows/columns p and q mix pairwise.
	for k := 0; k < n; k++ {
		if k == p || k == q {
			continue
		}

		akp, akq := a.At(k, p), a.At(k, q)
		a.Set(k, p, c*akp-s*akq)
		a.Set(p, k, a.At(k, p))
		a.Set(k, q, s*akp+c*akq)
		a.Set(q, k, a.At(k, q))
	}

	for k := 0; k < n; k++ {
		vkp, vkq := v.At(k, p), v.At(k, q)
		v.Set(k, p, c*vkp-s*vkq)
		v.Set(k, q, s*vkp+c*vkq)
	}
}
