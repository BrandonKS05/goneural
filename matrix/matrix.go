// Package matrix provides the dense float64 matrix type used throughout
// goneural. Elements live in a single flat row-major slice, so every
// operation costs one allocation and walks memory sequentially, instead of
// the slice-per-row layout (and per-element closure calls) of the
// github.com/BrandonKS05/goneural/matrix package this replaces.
//
// All operations return a new Matrix and never modify their receiver; the
// only mutating methods are Randomize, Zero and Set, which use pointer
// receivers. Shape mismatches panic, since they are programmer errors that
// have no meaningful runtime recovery.
package matrix

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// Mapper is the function type passed to Map: it receives an element's value
// and its row/column position and returns the replacement value.
type Mapper func(val float64, x, y int) float64

// Folder is the function type passed to Fold: it merges an element (and its
// row/column position) into the running accumulator.
type Folder func(accumulator, val float64, x, y int) float64

// Matrix is a dense row-major matrix. The zero value is an empty 0x0
// matrix. Rows and Columns are exported so shape checks read naturally
// (m.Rows); treat them as read-only, since reassigning them does not
// resize the backing data.
type Matrix struct {
	Rows    int
	Columns int
	data    []float64
}

func zeros(r, c int) Matrix {
	return Matrix{Rows: r, Columns: c, data: make([]float64, r*c)}
}

// New returns an r x c matrix filled row by row from data, or an all-zero
// matrix when data is nil. The data is copied, never aliased. It panics if
// data's shape doesn't match r and c.
func New(r, c int, data [][]float64) Matrix {
	if r < 0 || c < 0 {
		panic(fmt.Sprintf("matrix: New called with negative shape %dx%d", r, c))
	}

	m := zeros(r, c)
	if data == nil {
		return m
	}

	if len(data) != r {
		panic(fmt.Sprintf("matrix: New given %d rows of data, want %d", len(data), r))
	}
	for i, row := range data {
		if len(row) != c {
			panic(fmt.Sprintf("matrix: New row %d has %d values, want %d", i, len(row), c))
		}
		copy(m.data[i*c:(i+1)*c], row)
	}

	return m
}

// Zeros returns an r x c matrix with every element set to 0.
func Zeros(r, c int) Matrix {
	return New(r, c, nil)
}

// NewFromArray returns d as a column vector: len(d) rows and one column.
// The data is copied.
func NewFromArray(d []float64) Matrix {
	m := zeros(len(d), 1)
	copy(m.data, d)
	return m
}

// Unflatten builds an r x c matrix from a row-major flat slice, copying it.
// It is the inverse of Flatten.
func Unflatten(r, c int, data []float64) Matrix {
	if len(data) != r*c {
		panic(fmt.Sprintf("matrix: Unflatten given %d values for a %dx%d matrix", len(data), r, c))
	}

	m := zeros(r, c)
	copy(m.data, data)
	return m
}

// At returns the element at row i, column j.
func (m Matrix) At(i, j int) float64 {
	m.checkIndex(i, j)
	return m.data[i*m.Columns+j]
}

// Set assigns the element at row i, column j.
func (m *Matrix) Set(i, j int, v float64) {
	m.checkIndex(i, j)
	m.data[i*m.Columns+j] = v
}

func (m Matrix) checkIndex(i, j int) {
	if i < 0 || i >= m.Rows || j < 0 || j >= m.Columns {
		panic(fmt.Sprintf("matrix: index (%d, %d) out of range for %dx%d matrix", i, j, m.Rows, m.Columns))
	}
}

// Flatten returns the elements as a fresh row-major slice.
func (m Matrix) Flatten() []float64 {
	out := make([]float64, len(m.data))
	copy(out, m.data)
	return out
}

// Copy returns a deep copy of the matrix.
func (m Matrix) Copy() Matrix {
	return Unflatten(m.Rows, m.Columns, m.data)
}

// Randomize sets every element to an independent uniformly distributed
// random value in [low, high).
func (m *Matrix) Randomize(low, high float64) {
	for i := range m.data {
		m.data[i] = low + rand.Float64()*(high-low)
	}
}

// Zero sets every element to 0.
func (m *Matrix) Zero() {
	for i := range m.data {
		m.data[i] = 0
	}
}

// Map applies f to every element and returns the result as a new matrix.
func (m Matrix) Map(f Mapper) Matrix {
	out := zeros(m.Rows, m.Columns)
	idx := 0
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			out.data[idx] = f(m.data[idx], i, j)
			idx++
		}
	}
	return out
}

// Fold reduces the matrix to a single value by feeding every element
// through f along with the running accumulator.
func (m Matrix) Fold(f Folder, accumulator float64) float64 {
	idx := 0
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Columns; j++ {
			accumulator = f(accumulator, m.data[idx], i, j)
			idx++
		}
	}
	return accumulator
}

// Equal reports whether m and n have the same shape and exactly equal
// elements.
func (m Matrix) Equal(n Matrix) bool {
	if m.Rows != n.Rows || m.Columns != n.Columns {
		return false
	}
	for i, v := range m.data {
		if v != n.data[i] {
			return false
		}
	}
	return true
}

// ApproxEqual reports whether m and n have the same shape and every pair of
// elements is within eps of each other.
func (m Matrix) ApproxEqual(n Matrix, eps float64) bool {
	if m.Rows != n.Rows || m.Columns != n.Columns {
		return false
	}
	for i, v := range m.data {
		if math.Abs(v-n.data[i]) > eps {
			return false
		}
	}
	return true
}

func (m Matrix) String() string {
	var b strings.Builder
	for i := 0; i < m.Rows; i++ {
		b.WriteString("[")
		for j := 0; j < m.Columns; j++ {
			if j > 0 {
				b.WriteString(" ")
			}
			fmt.Fprintf(&b, "%g", m.data[i*m.Columns+j])
		}
		b.WriteString("]\n")
	}
	return b.String()
}
