package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// ClipByGlobalNorm rescales a set of matrices *jointly* so that the
// Frobenius norm of the whole collection -- every element of every matrix
// flattened into one vector -- is at most maxNorm; if it already is, the
// matrices come back unchanged. Scaling everything by one shared factor
// preserves the direction of the overall gradient, unlike clipping each
// element or each layer on its own, which would bend it; that makes this
// the standard defence against exploding gradients. A maxNorm of 0 or less
// disables clipping and returns the input as-is.
func ClipByGlobalNorm(ms []matrix.Matrix, maxNorm float64) []matrix.Matrix {
	if maxNorm <= 0 {
		return ms
	}

	sumSquares := 0.0
	for _, m := range ms {
		n := m.Norm()
		sumSquares += n * n
	}

	globalNorm := math.Sqrt(sumSquares)
	if globalNorm <= maxNorm {
		return ms
	}

	factor := maxNorm / globalNorm
	out := make([]matrix.Matrix, len(ms))
	for i, m := range ms {
		out[i] = m.Scale(factor)
	}
	return out
}
