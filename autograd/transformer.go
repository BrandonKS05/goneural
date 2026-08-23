package autograd

import (
	"math"
	"math/rand"

	"github.com/BrandonKS05/goneural/matrix"
)

// MultiHeadAttention runs several attention heads over the same input and
// concatenates what they retrieve.
//
// One head computes a single distribution per position, so it can look in
// exactly one place at a time -- and since that distribution is a softmax,
// attending to two things at once means attending to each of them at half
// strength, blending their values into something that is neither. Several
// heads, each with its own small query/key projection, sidestep the
// constraint: one can track syntactic position while another tracks
// content, and the output projection recombines them. Splitting a fixed
// model width across h heads costs nothing in parameters -- it buys
// independent attention patterns with the same arithmetic.
type MultiHeadAttention struct {
	// Query, Key and Value hold one projection per head, each of shape
	// head x model.
	Query []*Node
	Key   []*Node
	Value []*Node

	// Out projects the concatenated heads back to the model width.
	Out *Linear

	scale float64
}

// NewMultiHeadAttention builds heads attention heads over a model
// dimension of model. The per-head width is model/heads, so the
// concatenation comes back out at exactly model wide; model must divide
// evenly by heads.
func NewMultiHeadAttention(model, heads int) *MultiHeadAttention {
	if model < 1 || heads < 1 {
		panic("autograd: attention needs positive dimensions")
	}
	if model%heads != 0 {
		panic("autograd: the model dimension must divide evenly among the heads")
	}

	head := model / heads
	random := func(rows, cols int, name string) *Node {
		limit := math.Sqrt(6 / float64(rows+cols))
		m := matrix.New(rows, cols, nil).Map(func(_ float64, _, _ int) float64 {
			return (rand.Float64()*2 - 1) * limit
		})
		return Param(m).Named(name)
	}

	a := &MultiHeadAttention{
		Out:   NewLinear(model, model),
		scale: 1 / math.Sqrt(float64(head)),
	}

	for h := 0; h < heads; h++ {
		a.Query = append(a.Query, random(head, model, "mha.query"))
		a.Key = append(a.Key, random(head, model, "mha.key"))
		a.Value = append(a.Value, random(head, model, "mha.value"))
	}

	return a
}

// Heads reports how many heads this layer runs.
func (a *MultiHeadAttention) Heads() int {
	return len(a.Query)
}

// Parameters returns every learnable node, heads first.
func (a *MultiHeadAttention) Parameters() []*Node {
	params := make([]*Node, 0, 3*len(a.Query)+2)
	params = append(params, a.Query...)
	params = append(params, a.Key...)
	params = append(params, a.Value...)
	return append(params, a.Out.Parameters()...)
}

// Forward runs every head over an input of shape model x positions and
// projects the concatenation back to model x positions. mask, when
// non-nil, is added to every head's scores before their softmax (see
// CausalMask).
func (a *MultiHeadAttention) Forward(x *Node, mask *Node) *Node {
	outputs := make([]*Node, len(a.Query))
	for h := range a.Query {
		v := MatMul(a.Value[h], x)
		outputs[h] = MatMul(v, a.weights(h, x, mask))
	}

	return a.Out.Forward(ConcatRows(outputs...))
}

// Weights returns one head's attention distribution, of shape
// positions x positions, where column j is what position j looked at.
// Comparing heads on the same input is the usual way to see whether they
// specialized or collapsed onto the same pattern.
func (a *MultiHeadAttention) Weights(head int, x *Node, mask *Node) matrix.Matrix {
	if head < 0 || head >= len(a.Query) {
		panic("autograd: attention head index out of range")
	}
	return a.weights(head, x, mask).Value
}

func (a *MultiHeadAttention) weights(head int, x *Node, mask *Node) *Node {
	if x.Value.Rows != a.Query[head].Value.Columns {
		panic("autograd: attention input does not match the model dimension")
	}

	q := MatMul(a.Query[head], x)
	k := MatMul(a.Key[head], x)

	scores := Scale(MatMul(Transpose(k), q), a.scale)
	if mask != nil {
		if mask.Value.Rows != scores.Value.Rows || mask.Value.Columns != scores.Value.Columns {
			panic("autograd: attention mask must be positions x positions")
		}
		scores = Add(scores, mask)
	}

	return Softmax(scores)
}

// TransformerBlock is one layer of a transformer: multi-head attention
// followed by a position-wise feed-forward network, each wrapped in a
// residual connection with normalization in front.
//
// The two halves do complementary work. Attention is the only part that
// moves information *between* positions, but it moves it linearly -- an
// output is a weighted average of values, and a weighted average of
// vectors cannot compute much on its own. The feed-forward network then
// works on each position separately and non-linearly, which is where the
// retrieved information is actually processed. Neither is useful without
// the other.
//
// Normalization goes before each sub-layer rather than after it (the
// "pre-norm" arrangement). That leaves the residual path completely
// unobstructed from input to output, so a gradient can reach the earliest
// layer without passing through a single normalization -- which is what
// makes deep stacks trainable without a learning-rate warmup.
type TransformerBlock struct {
	Attention *MultiHeadAttention
	AttnNorm  *LayerNorm
	FFNorm    *LayerNorm

	// The feed-forward pair, conventionally widening by 4x in the middle:
	// it is where most of a transformer's parameters live and, on the
	// usual reading, where per-position knowledge is stored.
	Up   *Linear
	Down *Linear

	// Dropout is the rate applied to each sub-layer's output before it
	// rejoins the residual stream. It only fires when Forward is called
	// with training set.
	Dropout float64
}

// NewTransformerBlock builds a block over the given model width with the
// given number of heads and a feed-forward width of 4 * model.
func NewTransformerBlock(model, heads int) *TransformerBlock {
	return &TransformerBlock{
		Attention: NewMultiHeadAttention(model, heads),
		AttnNorm:  NewLayerNorm(model),
		FFNorm:    NewLayerNorm(model),
		Up:        NewLinear(model, 4*model),
		Down:      NewLinear(4*model, model),
	}
}

// Parameters returns every learnable node in the block.
func (b *TransformerBlock) Parameters() []*Node {
	params := b.Attention.Parameters()
	params = append(params, b.AttnNorm.Parameters()...)
	params = append(params, b.FFNorm.Parameters()...)
	params = append(params, b.Up.Parameters()...)
	return append(params, b.Down.Parameters()...)
}

// Forward runs the block over an input of shape model x positions.
func (b *TransformerBlock) Forward(x *Node, mask *Node, training bool) *Node {
	attended := b.Attention.Forward(b.AttnNorm.Forward(x), mask)
	x = Add(x, Dropout(attended, b.Dropout, training))

	wide := GELU(b.Up.Forward(b.FFNorm.Forward(x)))
	return Add(x, Dropout(b.Down.Forward(wide), b.Dropout, training))
}

// Embedding is a lookup table of one column per vocabulary entry, which is
// how discrete tokens enter a model that only understands vectors. It is
// mathematically a matrix multiply against a one-hot vector, implemented
// as a lookup because all but one term of that product is zero -- and the
// gradient, correspondingly, is scattered back only to the columns that
// were actually used.
type Embedding struct {
	Table *Node
}

// NewEmbedding builds a table of vocab entries, each a vector of the given
// width, initialized with small noise -- large embeddings would dominate
// the residual stream before training has any say in it.
func NewEmbedding(vocab, width int) *Embedding {
	if vocab < 1 || width < 1 {
		panic("autograd: Embedding needs positive dimensions")
	}

	table := matrix.New(width, vocab, nil).Map(func(_ float64, _, _ int) float64 {
		return rand.NormFloat64() * 0.02
	})

	return &Embedding{Table: Param(table).Named("embedding.table")}
}

// Parameters returns the learnable table.
func (e *Embedding) Parameters() []*Node {
	return []*Node{e.Table}
}

// Forward looks up each id, returning a width x len(ids) matrix -- one
// column per position, in order.
func (e *Embedding) Forward(ids []int) *Node {
	if len(ids) == 0 {
		panic("autograd: Embedding needs at least one id")
	}

	width, vocab := e.Table.Value.Rows, e.Table.Value.Columns
	value := matrix.New(width, len(ids), nil)

	for j, id := range ids {
		if id < 0 || id >= vocab {
			panic("autograd: Embedding id out of range")
		}
		for i := 0; i < width; i++ {
			value.Set(i, j, e.Table.Value.At(i, id))
		}
	}

	table := e.Table
	lookups := append([]int(nil), ids...)

	return newNode(value, func(grad matrix.Matrix) {
		scattered := matrix.New(width, vocab, nil)
		for j, id := range lookups {
			for i := 0; i < width; i++ {
				// Accumulated, not assigned: a token appearing twice in one
				// sequence must collect gradient from both positions.
				scattered.Set(i, id, scattered.At(i, id)+grad.At(i, j))
			}
		}
		push(table, scattered)
	}, table)
}

// PositionalEncoding returns the fixed width x positions matrix of
// sinusoids from Vaswani et al. (2017), to be added to a sequence of
// embeddings.
//
// Attention is entirely order-blind -- shuffle the input columns and the
// outputs shuffle with them, unchanged -- so position has to be written
// into the vectors themselves. Sinusoids of geometrically spaced
// frequencies do it without any parameters, and with the property that the
// encoding of position p+k is a fixed linear function of the encoding of
// p, which puts relative offsets within easy reach of a linear projection.
func PositionalEncoding(width, positions int) *Node {
	if width < 1 || positions < 1 {
		panic("autograd: PositionalEncoding needs positive dimensions")
	}

	m := matrix.New(width, positions, nil)
	for pos := 0; pos < positions; pos++ {
		for i := 0; i < width; i++ {
			// Each pair of rows shares a frequency, one as sine and one as
			// cosine; wavelengths run from 2*pi up to 10000*2*pi.
			frequency := 1 / math.Pow(10000, float64(2*(i/2))/float64(width))
			angle := float64(pos) * frequency

			if i%2 == 0 {
				m.Set(i, pos, math.Sin(angle))
			} else {
				m.Set(i, pos, math.Cos(angle))
			}
		}
	}

	return Const(m)
}
