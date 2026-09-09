package autograd

import (
	"github.com/BrandonKS05/goneural/matrix"
)

// GRUCell is one step of a Gated Recurrent Unit network (Cho et al., 2014).
//
// The GRU is what remains of the LSTM after asking which parts pay their
// way. The separate cell state is gone -- the hidden state itself is the
// memory highway -- and the forget and input gates are fused into one
// update gate: whatever fraction of the old state is kept, exactly the
// complement is written, so the new state is a convex blend of old state
// and candidate,
//
//	state' = update (.) state + (1 - update) (.) candidate
//
// and can never grow without bound the way an uncoupled write could make
// it. Where the update gate sits near 1 the state (and its gradient)
// passes through the step essentially untouched, which is the same
// vanishing-gradient cure the LSTM's additive cell provides.
//
// The LSTM's freedom to keep *and* write at once is genuinely lost, along
// with the output gate's ability to hold private state back from the next
// layer. Empirically the loss rarely shows: with a quarter fewer
// parameters for the same widths, neither architecture reliably beats the
// other (Greff et al., 2017; Jozefowicz et al., 2015).
//
// Note the convention above -- some texts flip the update gate's role and
// write state' = (1 - z) state + z candidate. Here update measures how
// much *survives*, so an open gate means remembering, matching the
// LSTM's forget gate.
type GRUCell struct {
	// Gates maps [state; input] to the update and reset pre-activations
	// stacked vertically, in that order.
	Gates *Linear

	// Candidate maps [reset (.) state; input] to the proposed new state.
	// Unlike the LSTM's four transformations, it cannot share the gates'
	// multiply: its input depends on the reset gate, which is itself the
	// gates' output.
	Candidate *Linear

	hidden int
}

// NewGRUCell builds a cell over inputs of the given width with a state of
// the given width.
//
// The update gate's bias starts at 1 rather than 0 -- the LSTM's
// forget-gate trick transplanted. A zero bias puts the gate at 0.5, so the
// cell halves what it carries at every step and has forgotten almost
// everything within ten; starting the gate open lets it remember by
// default and learn what to discard. Jozefowicz et al. (2015) found this
// one bias mattered more than most of the architectural variations they
// searched over.
func NewGRUCell(input, hidden int) *GRUCell {
	if input < 1 || hidden < 1 {
		panic("autograd: GRUCell needs positive dimensions")
	}

	gates := NewLinear(hidden+input, 2*hidden)
	for i := 0; i < hidden; i++ {
		gates.Bias.Value.Set(i, 0, 1) // the update gate's slice
	}

	return &GRUCell{
		Gates:     gates,
		Candidate: NewLinear(hidden+input, hidden),
		hidden:    hidden,
	}
}

// Hidden returns the state width.
func (c *GRUCell) Hidden() int {
	return c.hidden
}

// Parameters returns the learnable nodes.
func (c *GRUCell) Parameters() []*Node {
	return append(c.Gates.Parameters(), c.Candidate.Parameters()...)
}

// ZeroState returns the state a sequence starts from, with one column per
// sequence being processed in parallel. There is no LSTMState analogue to
// carry -- collapsing the state pair to one vector is half of what the GRU
// simplified away.
func (c *GRUCell) ZeroState(batch int) *Node {
	if batch < 1 {
		panic("autograd: GRUCell needs a positive batch size")
	}

	return Const(matrix.New(c.hidden, batch, nil))
}

// Step advances the cell by one input, returning the new state. x is
// input x batch and must share the state's batch width.
func (c *GRUCell) Step(x, state *Node) *Node {
	if x.Value.Columns != state.Value.Columns {
		panic("autograd: GRUCell input and state disagree on the batch size")
	}

	gates := c.Gates.Forward(ConcatRows(state, x))
	update := Sigmoid(rows(gates, 0, c.hidden))
	reset := Sigmoid(rows(gates, c.hidden, c.hidden))

	// The candidate reads history through the reset gate, so it can
	// propose a fresh start (reset near 0 blinds it to the state) without
	// having to overwrite anything -- whether the proposal is *taken* is
	// the update gate's separate decision.
	candidate := Tanh(c.Candidate.Forward(ConcatRows(Mul(reset, state), x)))

	// update (.) state + (1 - update) (.) candidate, rearranged so no
	// all-ones matrix needs materializing.
	return Add(candidate, Mul(update, Sub(state, candidate)))
}

// Run applies the cell across a sequence of inputs, returning each step's
// state along with the final one. As with the LSTM, nothing here schedules
// backpropagation through time: the unrolled graph is just a long chain,
// and Backward walks it like any other.
func (c *GRUCell) Run(inputs []*Node, state *Node) ([]*Node, *Node) {
	outputs := make([]*Node, len(inputs))
	for i, x := range inputs {
		state = c.Step(x, state)
		outputs[i] = state
	}
	return outputs, state
}
