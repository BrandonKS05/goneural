package autograd

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// LSTMCell is one step of a Long Short-Term Memory recurrent network
// (Hochreiter and Schmidhuber, 1997).
//
// A plain recurrent network multiplies its state by the same matrix at
// every step, so after n steps the signal has been multiplied n times: it
// either vanishes or explodes, and gradients do the same, which is why
// plain RNNs cannot learn dependencies more than a few steps apart. The
// LSTM's answer is a cell state that is *added* to rather than multiplied
// through -- a highway along which information (and gradient) can travel
// many steps unchanged -- guarded by three learned gates:
//
//   - the forget gate decides what to drop from the cell,
//   - the input gate decides what of the new candidate to write,
//   - the output gate decides how much of the cell to expose as the state.
//
// All four transformations (three gates and the candidate) read the same
// concatenation of the previous state and the current input, so they are
// computed as one matrix multiply and then sliced.
//
// Nothing here is unrolled or scheduled: applying the cell repeatedly
// builds a longer graph, and Backward walks it in reverse. Backpropagation
// through time is not implemented anywhere in this package -- it is what
// the general engine does when the graph happens to be a chain.
type LSTMCell struct {
	// Gates maps [state; input] to the four pre-activations stacked
	// vertically, in the order forget, input, candidate, output.
	Gates *Linear

	hidden int
}

// NewLSTMCell builds a cell over inputs of the given width with a state of
// the given width.
//
// The forget gate's bias starts at 1 rather than 0, which is a small
// detail with an outsized effect: a zero bias puts the forget gate at 0.5,
// so the cell halves its memory at every step and has forgotten almost
// everything within ten. Starting the gate open lets the cell remember by
// default and learn what to discard, rather than the reverse.
func NewLSTMCell(input, hidden int) *LSTMCell {
	if input < 1 || hidden < 1 {
		panic("autograd: LSTMCell needs positive dimensions")
	}

	gates := NewLinear(hidden+input, 4*hidden)
	for i := 0; i < hidden; i++ {
		gates.Bias.Value.Set(i, 0, 1) // the forget gate's slice
	}

	return &LSTMCell{Gates: gates, hidden: hidden}
}

// Hidden returns the state width.
func (c *LSTMCell) Hidden() int {
	return c.hidden
}

// Parameters returns the learnable nodes.
func (c *LSTMCell) Parameters() []*Node {
	return c.Gates.Parameters()
}

// LSTMState is the pair a cell carries between steps: the exposed hidden
// state and the internal cell state.
type LSTMState struct {
	Hidden *Node
	Cell   *Node
}

// ZeroState returns the state a sequence starts from, with one column per
// sequence being processed in parallel.
func (c *LSTMCell) ZeroState(batch int) LSTMState {
	if batch < 1 {
		panic("autograd: LSTMCell needs a positive batch size")
	}

	return LSTMState{
		Hidden: Const(matrix.New(c.hidden, batch, nil)),
		Cell:   Const(matrix.New(c.hidden, batch, nil)),
	}
}

// Step advances the cell by one input, returning the new state. x is
// input x batch and must share the state's batch width.
func (c *LSTMCell) Step(x *Node, state LSTMState) LSTMState {
	if x.Value.Columns != state.Hidden.Value.Columns {
		panic("autograd: LSTMCell input and state disagree on the batch size")
	}

	// One multiply for all four transformations, then sliced apart.
	gates := c.Gates.Forward(ConcatRows(state.Hidden, x))

	forget := Sigmoid(rows(gates, 0, c.hidden))
	input := Sigmoid(rows(gates, c.hidden, c.hidden))
	candidate := Tanh(rows(gates, 2*c.hidden, c.hidden))
	output := Sigmoid(rows(gates, 3*c.hidden, c.hidden))

	// The additive cell update: what survives the forget gate, plus what
	// the input gate admits. Because this is an addition rather than a
	// matrix multiply, the gradient along the cell path is scaled by the
	// forget gate alone -- near 1 when the gate is open, which is what
	// keeps it alive over long sequences.
	cell := Add(Mul(forget, state.Cell), Mul(input, candidate))

	return LSTMState{
		Hidden: Mul(output, Tanh(cell)),
		Cell:   cell,
	}
}

// Run applies the cell across a sequence of inputs, returning each step's
// hidden state along with the final state. The graph it builds is as long
// as the sequence, and differentiating it is ordinary backpropagation over
// that graph.
func (c *LSTMCell) Run(inputs []*Node, state LSTMState) ([]*Node, LSTMState) {
	outputs := make([]*Node, len(inputs))
	for i, x := range inputs {
		state = c.Step(x, state)
		outputs[i] = state.Hidden
	}
	return outputs, state
}

// rows slices count rows out of a node, starting at offset -- the inverse
// of ConcatRows, and how the four gate transformations are separated from
// the single multiply that produced them.
func rows(x *Node, offset, count int) *Node {
	if offset < 0 || count < 1 || offset+count > x.Value.Rows {
		panic("autograd: row slice out of range")
	}

	value := matrix.New(count, x.Value.Columns, nil)
	for i := 0; i < count; i++ {
		for j := 0; j < x.Value.Columns; j++ {
			value.Set(i, j, x.Value.At(offset+i, j))
		}
	}

	total, columns := x.Value.Rows, x.Value.Columns

	return newNode(value, func(grad matrix.Matrix) {
		// The rows that were not selected contributed nothing, so their
		// gradient is zero; the selected ones get theirs back in place.
		full := matrix.New(total, columns, nil)
		for i := 0; i < count; i++ {
			for j := 0; j < columns; j++ {
				full.Set(offset+i, j, grad.At(i, j))
			}
		}
		push(x, full)
	}, x)
}

// Rows exposes the row slice as a general operation, since selecting part
// of a wider representation is useful well beyond the gates it was written
// for.
func Rows(x *Node, offset, count int) *Node {
	return rows(x, offset, count)
}

// ClipGradients rescales the parameters' gradients so their combined
// Euclidean norm is at most limit, leaving them untouched when they are
// already below it.
//
// Recurrent models need this more than anything else here. A long
// unrolled graph occasionally produces one enormous gradient -- a single
// unlucky step where the gates line up -- and one such step can undo
// thousands of good ones by throwing the weights somewhere useless.
// Clipping the whole set together, rather than each tensor separately,
// preserves the gradient's direction and rescales only its length.
//
// It returns the norm before clipping, which is worth logging: a norm that
// spikes by orders of magnitude is the clearest early sign a model is
// about to diverge.
func ClipGradients(limit float64, params ...*Node) float64 {
	if limit <= 0 {
		panic("autograd: gradient clipping needs a positive limit")
	}

	total := 0.0
	for _, p := range params {
		for _, g := range p.Grad.Flatten() {
			total += g * g
		}
	}

	norm := math.Sqrt(total)
	if norm <= limit || norm == 0 {
		return norm
	}

	scale := limit / norm
	for _, p := range params {
		p.Grad = p.Grad.Scale(scale)
	}

	return norm
}
