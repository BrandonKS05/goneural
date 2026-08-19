// Package autograd is a small reverse-mode automatic differentiation
// engine over matrices.
//
// The rest of this module hard-codes one architecture: a stack of dense
// layers, whose gradient is a backprop recurrence written out by hand in
// gradients.go. That is fast and clear, and it is also the reason every
// activation in the package must be invertible from its own output, why
// softmax needs a special case in three separate optimizers, and why there
// is no way to express a network that is not a straight chain of matrix
// multiplies.
//
// This package removes that ceiling. Every operation records how it was
// computed, so a program built out of these operations *is* its own
// derivative: build any expression, call Backward on the scalar at the
// end, and read the gradient off each input. Nothing here knows what a
// neural network is -- which is exactly why attention.go can build a
// transformer-style attention head on top of it without this file changing
// at all.
//
// The mechanism is the chain rule plus bookkeeping. Each Node remembers
// its parents and a closure that, given the gradient flowing into it,
// accumulates the corresponding gradient into those parents. Backward
// walks the graph in reverse topological order -- every node is visited
// only after everything that consumes it -- so each node's gradient is
// complete before it is propagated further back.
package autograd

import (
	"github.com/BrandonKS05/goneural/matrix"
)

// Node is one value in a computation graph: the matrix that was computed,
// the gradient of the final scalar with respect to it, and the record of
// where it came from.
//
// Nodes are built by the operations in ops.go and by Param/Const; the zero
// Node is not useful. They are not safe for concurrent use, since a
// backward pass writes gradients into every node reachable from the root.
type Node struct {
	// Value is what forward evaluation produced.
	Value matrix.Matrix

	// Grad is d(root)/d(this), accumulated during Backward. It is
	// meaningful only for nodes reachable from the node Backward was
	// called on, and only until the next ZeroGrad.
	Grad matrix.Matrix

	// Name is an optional label, used only in error messages and when
	// printing a graph.
	Name string

	// requiresGrad marks the leaves worth differentiating -- the
	// parameters. A leaf without it (a constant, an input batch) still
	// propagates gradients through itself if it has parents, but as a leaf
	// it accumulates nothing, which is what keeps a large fixed input from
	// carrying a gradient matrix around for no reason.
	requiresGrad bool

	parents  []*Node
	backward func(grad matrix.Matrix)
}

// Param wraps a matrix as a differentiable leaf: a weight, a bias,
// anything an optimizer should update. Its Grad is filled in by Backward.
func Param(value matrix.Matrix) *Node {
	return &Node{
		Value:        value,
		Grad:         matrix.New(value.Rows, value.Columns, nil),
		requiresGrad: true,
	}
}

// Const wraps a matrix as a non-differentiable leaf: an input batch, a
// target, a fixed mask. Gradients flow *through* expressions built on it,
// but nothing accumulates on the constant itself.
func Const(value matrix.Matrix) *Node {
	return &Node{Value: value}
}

// Named returns the node with a label attached, for readable errors and
// graph dumps. It mutates and returns the receiver, so it chains:
//
//	w := autograd.Param(weights).Named("layer1.w")
func (n *Node) Named(name string) *Node {
	n.Name = name
	return n
}

// RequiresGrad reports whether this node is a differentiable leaf.
func (n *Node) RequiresGrad() bool {
	return n.requiresGrad
}

// Item returns the value of a 1x1 node, which is what losses and other
// reductions produce. It panics on any other shape.
func (n *Node) Item() float64 {
	if n.Value.Rows != 1 || n.Value.Columns != 1 {
		panic("autograd: Item on a non-scalar node")
	}
	return n.Value.At(0, 0)
}

// newNode builds an interior node: the result of an operation, wired to
// the parents it was computed from and to the closure that pushes gradient
// back into them.
func newNode(value matrix.Matrix, backward func(grad matrix.Matrix), parents ...*Node) *Node {
	return &Node{
		Value:    value,
		Grad:     matrix.New(value.Rows, value.Columns, nil),
		parents:  parents,
		backward: backward,
	}
}

// accumulate adds a gradient contribution into a node. Addition, not
// assignment, is the entire reason a value used twice in an expression
// gets the sum of both paths' gradients -- the multivariable chain rule,
// falling out of the bookkeeping.
func (n *Node) accumulate(grad matrix.Matrix) {
	if grad.Rows != n.Value.Rows || grad.Columns != n.Value.Columns {
		panic("autograd: gradient shape does not match the value it belongs to")
	}
	if n.Grad.Rows == 0 {
		n.Grad = matrix.New(n.Value.Rows, n.Value.Columns, nil)
	}
	n.Grad = n.Grad.AddMatrix(grad)
}

// Backward computes the gradient of this node with respect to everything
// it was computed from, filling in each reachable node's Grad. The node
// must be a scalar: a gradient is only well defined per output, and every
// training objective ends in one number anyway.
//
// Leaf gradients accumulate rather than overwrite, so running Backward
// over several mini-batches before stepping sums their gradients -- the
// standard way to train on a batch larger than memory allows. Interior
// gradients are cleared at the start of each pass instead: they are
// scratch space for the walk, and leaving them in place would make a
// second pass propagate through its own leftovers and compound.
func (n *Node) Backward() {
	if n.Value.Rows != 1 || n.Value.Columns != 1 {
		panic("autograd: Backward on a non-scalar node")
	}

	order := n.topological()

	for _, node := range order {
		if node.backward != nil {
			node.Grad = matrix.New(node.Value.Rows, node.Value.Columns, nil)
		}
	}

	// Seed the root: d(root)/d(root) is 1.
	n.accumulate(matrix.New(1, 1, [][]float64{{1}}))

	// Reverse topological order guarantees a node is processed only once
	// every consumer of it has already pushed its share of the gradient in,
	// so n.Grad is final by the time it is passed on.
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		if node.backward != nil {
			node.backward(node.Grad)
		}
	}
}

// topological returns every node reachable from n, parents before
// children. The walk is iterative rather than recursive: a deep graph (a
// long unrolled sequence, say) would otherwise be limited by the stack.
func (n *Node) topological() []*Node {
	var order []*Node
	seen := map[*Node]bool{}

	type frame struct {
		node     *Node
		expanded bool
	}

	stack := []frame{{node: n}}
	for len(stack) > 0 {
		top := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if top.expanded {
			order = append(order, top.node)
			continue
		}
		if seen[top.node] {
			continue
		}
		seen[top.node] = true

		// Push the node back marked as expanded, so it lands in the output
		// only after everything it depends on has been emitted.
		stack = append(stack, frame{node: top.node, expanded: true})
		for _, parent := range top.node.parents {
			if !seen[parent] {
				stack = append(stack, frame{node: parent})
			}
		}
	}

	return order
}

// ZeroGrad clears the gradient of every node reachable from this one,
// which is what a training loop does between steps.
func (n *Node) ZeroGrad() {
	for _, node := range n.topological() {
		node.Grad = matrix.New(node.Value.Rows, node.Value.Columns, nil)
	}
}
