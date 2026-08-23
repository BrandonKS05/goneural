package autograd

import (
	"math"
	"math/rand"

	"github.com/BrandonKS05/goneural/matrix"
)

// This file holds the reusable layers a model is assembled from. Each is a
// plain struct of Param nodes plus a Forward that builds graph operations,
// so a model is just a function -- there is no layer interface to satisfy,
// no registry, and nothing that needs to know the shape of the network in
// advance.

// Linear is a dense layer: y = W x + b, with the batch (or sequence
// positions) across the columns of x.
type Linear struct {
	Weight *Node
	Bias   *Node
}

// NewLinear builds a dense layer mapping in features to out, with
// Xavier/Glorot-scaled weights and zero biases -- the initialization that
// keeps the variance of activations roughly constant from layer to layer,
// so a deep stack neither saturates nor fades on the first forward pass.
func NewLinear(in, out int) *Linear {
	if in < 1 || out < 1 {
		panic("autograd: Linear needs positive dimensions")
	}

	limit := math.Sqrt(6 / float64(in+out))
	weight := matrix.New(out, in, nil).Map(func(_ float64, _, _ int) float64 {
		return (rand.Float64()*2 - 1) * limit
	})

	return &Linear{
		Weight: Param(weight).Named("linear.weight"),
		Bias:   Param(matrix.New(out, 1, nil)).Named("linear.bias"),
	}
}

// Forward applies the layer.
func (l *Linear) Forward(x *Node) *Node {
	return AddBias(MatMul(l.Weight, x), l.Bias)
}

// Parameters returns the layer's learnable nodes.
func (l *Linear) Parameters() []*Node {
	return []*Node{l.Weight, l.Bias}
}

// LayerNorm normalizes each position independently: within one column, the
// features are shifted to zero mean and scaled to unit variance, then
// given a learned per-feature scale and shift.
//
// Note what it does *not* do -- it never looks across the batch. Batch
// normalization's statistics depend on which other samples happen to share
// the batch, which makes it awkward at inference, sensitive to batch size,
// and ill-defined for a single sequence. Normalizing within a position
// sidesteps all of that, which is why every transformer uses this one.
//
// Its real job is keeping a residual stack trainable: without it, the
// activations flowing through repeated additions grow without bound and
// the gradients follow.
type LayerNorm struct {
	Gain  *Node
	Shift *Node

	// Epsilon guards the division when a position's features are all
	// identical and the variance is zero.
	Epsilon float64
}

// NewLayerNorm builds a normalizer over the given feature count, starting
// from the identity transform (unit gain, zero shift) so it changes
// nothing until training gives it a reason to.
func NewLayerNorm(features int) *LayerNorm {
	if features < 1 {
		panic("autograd: LayerNorm needs a positive feature count")
	}

	gain := matrix.New(features, 1, nil).Map(func(_ float64, _, _ int) float64 { return 1 })

	return &LayerNorm{
		Gain:    Param(gain).Named("layernorm.gain"),
		Shift:   Param(matrix.New(features, 1, nil)).Named("layernorm.shift"),
		Epsilon: 1e-5,
	}
}

// Parameters returns the learnable nodes.
func (l *LayerNorm) Parameters() []*Node {
	return []*Node{l.Gain, l.Shift}
}

// Forward normalizes each column of x and applies the learned gain and
// shift.
//
// The normalization is one fused node rather than a composition of mean,
// variance and division ops. That is not only for speed: because the mean
// and variance are themselves functions of every element in the column,
// the derivative has two correction terms beyond the obvious one -- a
// change in any feature moves the statistics, which moves every *other*
// normalized feature. Writing it out once, and checking it against finite
// differences, is far safer than trusting a chain of broadcasts to
// reproduce it.
func (l *LayerNorm) Forward(x *Node) *Node {
	if x.Value.Rows != l.Gain.Value.Rows {
		panic("autograd: LayerNorm applied to the wrong feature count")
	}

	rows, columns := x.Value.Rows, x.Value.Columns
	n := float64(rows)

	normalized := matrix.New(rows, columns, nil)
	inverseDeviation := make([]float64, columns)

	for j := 0; j < columns; j++ {
		mean := 0.0
		for i := 0; i < rows; i++ {
			mean += x.Value.At(i, j)
		}
		mean /= n

		variance := 0.0
		for i := 0; i < rows; i++ {
			d := x.Value.At(i, j) - mean
			variance += d * d
		}
		variance /= n

		inverseDeviation[j] = 1 / math.Sqrt(variance+l.Epsilon)
		for i := 0; i < rows; i++ {
			normalized.Set(i, j, (x.Value.At(i, j)-mean)*inverseDeviation[j])
		}
	}

	value := normalized.Map(func(v float64, i, _ int) float64 {
		return v*l.Gain.Value.At(i, 0) + l.Shift.Value.At(i, 0)
	})

	gain, shift := l.Gain, l.Shift

	return newNode(value, func(grad matrix.Matrix) {
		// The learned parameters see every position, so their gradients
		// sum along the row.
		push(gain, grad.HadamardProduct(normalized).RowSums())
		push(shift, grad.RowSums())

		out := matrix.New(rows, columns, nil)
		for j := 0; j < columns; j++ {
			// dNormalized, and the two column-wide averages that account
			// for this column's mean and variance shifting when any single
			// element does.
			meanGrad, meanGradTimesNorm := 0.0, 0.0
			for i := 0; i < rows; i++ {
				g := grad.At(i, j) * gain.Value.At(i, 0)
				meanGrad += g
				meanGradTimesNorm += g * normalized.At(i, j)
			}
			meanGrad /= n
			meanGradTimesNorm /= n

			for i := 0; i < rows; i++ {
				g := grad.At(i, j) * gain.Value.At(i, 0)
				out.Set(i, j, inverseDeviation[j]*(g-meanGrad-normalized.At(i, j)*meanGradTimesNorm))
			}
		}
		push(x, out)
	}, x, gain, shift)
}

// Dropout zeroes each element with probability rate during training and
// scales the survivors by 1/(1-rate), so the expected activation is
// unchanged and nothing has to be rescaled at inference. Passing training
// as false returns x untouched, which is what a prediction path wants.
//
// It takes the rate and flag as arguments rather than storing them,
// because a dropout mask is per-forward-pass state, not a parameter: the
// same layer used twice in one graph must draw two independent masks.
func Dropout(x *Node, rate float64, training bool) *Node {
	if rate < 0 || rate >= 1 {
		panic("autograd: Dropout rate must be in [0, 1)")
	}
	if !training || rate == 0 {
		return x
	}

	keep := 1 - rate
	mask := x.Value.Map(func(_ float64, _, _ int) float64 {
		if rand.Float64() < keep {
			return 1 / keep
		}
		return 0
	})

	return newNode(x.Value.HadamardProduct(mask), func(grad matrix.Matrix) {
		// Gradient flows only where the forward pass did, through the same
		// mask and the same scaling.
		push(x, grad.HadamardProduct(mask))
	}, x)
}
