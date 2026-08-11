package matrix

import (
	"fmt"
	"math"
)

func (m Matrix) checkSameShape(n Matrix, op string) {
	if m.Rows != n.Rows || m.Columns != n.Columns {
		panic(fmt.Sprintf("matrix: %s needs equal shapes, got %dx%d and %dx%d", op, m.Rows, m.Columns, n.Rows, n.Columns))
	}
}

// AddMatrix returns the elementwise sum m + n.
func (m Matrix) AddMatrix(n Matrix) Matrix {
	m.checkSameShape(n, "AddMatrix")
	out := zeros(m.Rows, m.Columns)
	for i, v := range m.data {
		out.data[i] = v + n.data[i]
	}
	return out
}

// SubtractMatrix returns the elementwise difference m - n.
func (m Matrix) SubtractMatrix(n Matrix) Matrix {
	m.checkSameShape(n, "SubtractMatrix")
	out := zeros(m.Rows, m.Columns)
	for i, v := range m.data {
		out.data[i] = v - n.data[i]
	}
	return out
}

// HadamardProduct returns the elementwise product m * n.
func (m Matrix) HadamardProduct(n Matrix) Matrix {
	m.checkSameShape(n, "HadamardProduct")
	out := zeros(m.Rows, m.Columns)
	for i, v := range m.data {
		out.data[i] = v * n.data[i]
	}
	return out
}

// Add returns m with a added to every element.
func (m Matrix) Add(a float64) Matrix {
	out := zeros(m.Rows, m.Columns)
	for i, v := range m.data {
		out.data[i] = v + a
	}
	return out
}

// Subtract returns m with a subtracted from every element.
func (m Matrix) Subtract(a float64) Matrix {
	return m.Add(-a)
}

// Scale returns m with every element multiplied by a.
func (m Matrix) Scale(a float64) Matrix {
	out := zeros(m.Rows, m.Columns)
	for i, v := range m.data {
		out.data[i] = v * a
	}
	return out
}

// Divide returns m with every element divided by a.
func (m Matrix) Divide(a float64) Matrix {
	return m.Scale(1 / a)
}

// Sum returns the sum of all elements.
func (m Matrix) Sum() float64 {
	sum := 0.0
	for _, v := range m.data {
		sum += v
	}
	return sum
}

// Mean returns the average of all elements. It panics on an empty matrix,
// which has no meaningful mean.
func (m Matrix) Mean() float64 {
	if len(m.data) == 0 {
		panic("matrix: Mean of empty matrix")
	}
	return m.Sum() / float64(len(m.data))
}

// Min returns the smallest element. It panics on an empty matrix.
func (m Matrix) Min() float64 {
	if len(m.data) == 0 {
		panic("matrix: Min of empty matrix")
	}
	min := m.data[0]
	for _, v := range m.data[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

// Max returns the largest element. It panics on an empty matrix.
func (m Matrix) Max() float64 {
	if len(m.data) == 0 {
		panic("matrix: Max of empty matrix")
	}
	max := m.data[0]
	for _, v := range m.data[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// Trace returns the sum of the main diagonal. It panics if the matrix isn't
// square.
func (m Matrix) Trace() float64 {
	if m.Rows != m.Columns {
		panic(fmt.Sprintf("matrix: Trace of non-square %dx%d matrix", m.Rows, m.Columns))
	}
	sum := 0.0
	for i := 0; i < m.Rows; i++ {
		sum += m.data[i*m.Columns+i]
	}
	return sum
}

// Norm returns the Frobenius norm: the square root of the sum of every
// element squared. For a column vector this is the ordinary Euclidean
// length, which makes it the natural measure for gradient clipping.
func (m Matrix) Norm() float64 {
	sum := 0.0
	for _, v := range m.data {
		sum += v * v
	}
	return math.Sqrt(sum)
}

// Clip returns m with every element clamped into [low, high]. It panics if
// low is greater than high.
func (m Matrix) Clip(low, high float64) Matrix {
	if low > high {
		panic(fmt.Sprintf("matrix: Clip called with low %g > high %g", low, high))
	}

	out := zeros(m.Rows, m.Columns)
	for i, v := range m.data {
		out.data[i] = math.Min(math.Max(v, low), high)
	}
	return out
}

// Transpose returns the transposed matrix.
func (m Matrix) Transpose() Matrix {
	out := zeros(m.Columns, m.Rows)
	for i := 0; i < m.Rows; i++ {
		row := m.data[i*m.Columns : (i+1)*m.Columns]
		for j, v := range row {
			out.data[j*m.Rows+i] = v
		}
	}
	return out
}

// Multiply returns the matrix product m x n. It panics unless m.Columns
// equals n.Rows.
func (m Matrix) Multiply(n Matrix) Matrix {
	if m.Columns != n.Rows {
		panic(fmt.Sprintf("matrix: Multiply shape mismatch, %dx%d x %dx%d", m.Rows, m.Columns, n.Rows, n.Columns))
	}

	out := zeros(m.Rows, n.Columns)

	// Matrix-times-column-vector is the dominant shape in a neural network
	// forward/backward pass, and collapses to one dot product per row.
	if n.Columns == 1 {
		for i := 0; i < m.Rows; i++ {
			row := m.data[i*m.Columns : (i+1)*m.Columns]
			sum := 0.0
			for k, v := range row {
				sum += v * n.data[k]
			}
			out.data[i] = sum
		}
		return out
	}

	// i-k-j order keeps all three matrices walking rows sequentially, which
	// is far kinder to the cache than the naive i-j-k order.
	for i := 0; i < m.Rows; i++ {
		mRow := m.data[i*m.Columns : (i+1)*m.Columns]
		outRow := out.data[i*n.Columns : (i+1)*n.Columns]
		for k, mik := range mRow {
			nRow := n.data[k*n.Columns : (k+1)*n.Columns]
			for j, nkj := range nRow {
				outRow[j] += mik * nkj
			}
		}
	}
	return out
}

// Solve returns the x satisfying m * x = b, computed by Gauss-Jordan
// elimination with partial pivoting; b may hold several right-hand sides as
// columns. The second return value is false when m is singular (no unique
// solution exists). It panics unless m is square with as many rows as b.
func (m Matrix) Solve(b Matrix) (Matrix, bool) {
	if m.Rows != m.Columns {
		panic(fmt.Sprintf("matrix: Solve with non-square %dx%d coefficient matrix", m.Rows, m.Columns))
	}
	if m.Rows != b.Rows {
		panic(fmt.Sprintf("matrix: Solve shape mismatch, %dx%d coefficients with %dx%d right-hand side", m.Rows, m.Columns, b.Rows, b.Columns))
	}

	n := m.Rows
	a := m.Copy()
	x := b.Copy()

	for col := 0; col < n; col++ {
		// Pivot on the largest remaining entry in this column, for
		// numerical stability (same strategy as Det).
		pivot := col
		largest := math.Abs(a.data[col*n+col])
		for r := col + 1; r < n; r++ {
			if abs := math.Abs(a.data[r*n+col]); abs > largest {
				largest, pivot = abs, r
			}
		}
		if largest == 0 {
			return Matrix{}, false
		}
		if pivot != col {
			for k := 0; k < n; k++ {
				a.data[pivot*n+k], a.data[col*n+k] = a.data[col*n+k], a.data[pivot*n+k]
			}
			for k := 0; k < x.Columns; k++ {
				x.data[pivot*x.Columns+k], x.data[col*x.Columns+k] = x.data[col*x.Columns+k], x.data[pivot*x.Columns+k]
			}
		}

		p := a.data[col*n+col]
		for k := 0; k < n; k++ {
			a.data[col*n+k] /= p
		}
		for k := 0; k < x.Columns; k++ {
			x.data[col*x.Columns+k] /= p
		}

		// Eliminate the column everywhere else, reducing a to the identity
		// and x to the solution in one sweep.
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := a.data[r*n+col]
			if f == 0 {
				continue
			}
			for k := 0; k < n; k++ {
				a.data[r*n+k] -= f * a.data[col*n+k]
			}
			for k := 0; k < x.Columns; k++ {
				x.data[r*x.Columns+k] -= f * x.data[col*x.Columns+k]
			}
		}
	}

	return x, true
}

// Inverse returns the multiplicative inverse of m. The second return value
// is false when m is singular and no inverse exists. It panics if m isn't
// square.
func (m Matrix) Inverse() (Matrix, bool) {
	if m.Rows != m.Columns {
		panic(fmt.Sprintf("matrix: Inverse of non-square %dx%d matrix", m.Rows, m.Columns))
	}
	return m.Solve(Identity(m.Rows))
}

// Det returns the determinant, computed by Gaussian elimination with
// partial pivoting in O(n^3) time. It panics if the matrix isn't square.
// The determinant of the empty 0x0 matrix is 1 by convention.
func (m Matrix) Det() float64 {
	if m.Rows != m.Columns {
		panic(fmt.Sprintf("matrix: Det of non-square %dx%d matrix", m.Rows, m.Columns))
	}

	n := m.Rows
	a := m.Copy()
	det := 1.0

	for col := 0; col < n; col++ {
		// Pivot on the largest remaining entry in this column, for
		// numerical stability.
		pivot := col
		largest := math.Abs(a.data[col*n+col])
		for r := col + 1; r < n; r++ {
			if abs := math.Abs(a.data[r*n+col]); abs > largest {
				largest, pivot = abs, r
			}
		}
		if largest == 0 {
			return 0
		}
		if pivot != col {
			for k := col; k < n; k++ {
				a.data[pivot*n+k], a.data[col*n+k] = a.data[col*n+k], a.data[pivot*n+k]
			}
			det = -det
		}

		p := a.data[col*n+col]
		det *= p
		for r := col + 1; r < n; r++ {
			f := a.data[r*n+col] / p
			if f == 0 {
				continue
			}
			for k := col; k < n; k++ {
				a.data[r*n+k] -= f * a.data[col*n+k]
			}
		}
	}

	return det
}
