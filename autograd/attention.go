package autograd

import (
	"math"
	"math/rand"

	"github.com/BrandonKS05/goneural/matrix"
)

// AttentionHead is a single head of scaled dot-product self-attention --
// the operation transformers are built out of, and something the fixed
// dense-stack architecture in the parent package cannot express at all.
//
// A dense layer applies the same fixed weights to every input. Attention
// instead computes, for each position, *where to look*: every position
// emits a query, a key and a value; each query is scored against every key;
// the scores become a distribution through softmax; and the position's
// output is that distribution's weighted average of the values. The
// weights are therefore data-dependent -- a function of the input rather
// than of the parameters alone -- which is what lets one position pull in
// information from another based on content rather than on position.
//
// Sequences are laid out with the model dimension down and positions
// across, matching the batch convention in ops.go: an input of shape
// model x positions gives an output of the same shape. Everything here is
// composed from the ordinary operations in this package, so it
// differentiates itself; there is no attention-specific backward pass.
type AttentionHead struct {
	// Query, Key and Value project the input into the three roles.
	// Query and Key have shape head x model, Value has shape model x model
	// so the head's output can be added straight back onto its input.
	Query *Node
	Key   *Node
	Value *Node

	// scale is 1/sqrt(head), applied to the scores before the softmax. Dot
	// products of two independent head-dimensional vectors grow like
	// sqrt(head), and a softmax over large scores saturates into a hard
	// argmax whose gradient is nearly zero -- so without this the head
	// stops learning the moment it is made wide.
	scale float64
}

// NewAttentionHead builds a head over a model dimension of model, with
// queries and keys of width head. The projections are initialized with
// Xavier-scaled noise, which keeps the initial scores small enough for the
// softmax to stay in its responsive range.
func NewAttentionHead(model, head int) *AttentionHead {
	if model < 1 || head < 1 {
		panic("autograd: attention needs positive model and head dimensions")
	}

	random := func(rows, cols int) *Node {
		limit := math.Sqrt(6.0 / float64(rows+cols))
		m := matrix.New(rows, cols, nil)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				m.Set(i, j, (rand.Float64()*2-1)*limit)
			}
		}
		return Param(m)
	}

	return &AttentionHead{
		Query: random(head, model).Named("attention.query"),
		Key:   random(head, model).Named("attention.key"),
		Value: random(model, model).Named("attention.value"),
		scale: 1 / math.Sqrt(float64(head)),
	}
}

// Parameters returns the head's learnable nodes, ready to hand to an
// optimizer's Step.
func (a *AttentionHead) Parameters() []*Node {
	return []*Node{a.Query, a.Key, a.Value}
}

// Forward runs self-attention over a sequence of shape model x positions.
//
// mask, when non-nil, is added to the scores before the softmax and must
// be positions x positions. Since it is added *before* normalizing, a
// large negative entry drives that weight to essentially zero while
// leaving the remaining weights a valid distribution -- which is how a
// causal mask (see CausalMask) stops a position from attending to the
// future without any special handling in the softmax itself.
func (a *AttentionHead) Forward(input *Node, mask *Node) *Node {
	if input.Value.Rows != a.Value.Value.Columns {
		panic("autograd: attention input does not match the model dimension")
	}

	q := MatMul(a.Query, input) // head x positions
	k := MatMul(a.Key, input)   // head x positions
	v := MatMul(a.Value, input) // model x positions

	// scores[i][j] is how much position j's query matches position i's key,
	// so each *column* holds one query's scores over all keys -- which is
	// the axis Softmax normalizes.
	scores := Scale(MatMul(Transpose(k), q), a.scale)
	if mask != nil {
		if mask.Value.Rows != scores.Value.Rows || mask.Value.Columns != scores.Value.Columns {
			panic("autograd: attention mask must be positions x positions")
		}
		scores = Add(scores, mask)
	}

	weights := Softmax(scores) // positions x positions, columns sum to 1

	// Each output column is the value-weighted average its query asked for.
	return MatMul(v, weights)
}

// CausalMask returns the positions x positions additive mask that stops
// each position from attending to any later one: zero on and below the
// diagonal, a large negative number above it. This is the mask that makes
// attention usable for prediction -- without it, a model asked to predict
// the next token can simply read it.
func CausalMask(positions int) *Node {
	// Large but finite: an actual -Inf would produce NaN wherever a row is
	// entirely masked, and rounds to zero weight just the same.
	const negative = -1e9

	m := matrix.New(positions, positions, nil)
	for i := 0; i < positions; i++ {
		for j := 0; j < positions; j++ {
			if i > j { // key i is ahead of query j
				m.Set(i, j, negative)
			}
		}
	}

	return Const(m)
}
