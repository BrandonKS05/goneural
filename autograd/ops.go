package autograd

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// The operations below all follow the same shape: compute the forward
// value, then hand newNode a closure that takes the gradient flowing into
// the result and pushes each parent's share of it backward. Read any one
// of them as a line of the chain rule.
//
// The layout convention throughout is features down, batch across: a
// matrix of activations has one column per sample, matching the column
// vectors the rest of this module passes around. A dense layer is
// therefore MatMul(w, x) with w of shape out x in, and AddBias broadcasts
// one bias column across the whole batch.

// push routes a gradient contribution to a parent, skipping leaves that
// were declared constant -- there is nothing to accumulate on an input
// batch, and not walking into it saves the allocation.
func push(parent *Node, grad matrix.Matrix) {
	if parent.requiresGrad || parent.backward != nil {
		parent.accumulate(grad)
	}
}

// Add returns a + b elementwise. Addition splits an incoming gradient
// unchanged between both sides.
func Add(a, b *Node) *Node {
	checkSameShape(a, b, "Add")

	return newNode(a.Value.AddMatrix(b.Value), func(grad matrix.Matrix) {
		push(a, grad)
		push(b, grad)
	}, a, b)
}

// Sub returns a - b elementwise.
func Sub(a, b *Node) *Node {
	checkSameShape(a, b, "Sub")

	return newNode(a.Value.SubtractMatrix(b.Value), func(grad matrix.Matrix) {
		push(a, grad)
		push(b, grad.Scale(-1))
	}, a, b)
}

// Mul returns the elementwise (Hadamard) product. Each side's gradient is
// scaled by the *other* side's value, which is the product rule.
func Mul(a, b *Node) *Node {
	checkSameShape(a, b, "Mul")

	return newNode(a.Value.HadamardProduct(b.Value), func(grad matrix.Matrix) {
		push(a, grad.HadamardProduct(b.Value))
		push(b, grad.HadamardProduct(a.Value))
	}, a, b)
}

// MatMul returns the matrix product a . b. Its backward pass is the one
// worth memorizing: with C = A B, the gradient of A is G B^T and the
// gradient of B is A^T G -- each factor's gradient is the incoming one
// multiplied by the other factor, transposed to make the shapes meet.
func MatMul(a, b *Node) *Node {
	if a.Value.Columns != b.Value.Rows {
		panic("autograd: MatMul shape mismatch")
	}

	return newNode(a.Value.Multiply(b.Value), func(grad matrix.Matrix) {
		push(a, grad.Multiply(b.Value.Transpose()))
		push(b, a.Value.Transpose().Multiply(grad))
	}, a, b)
}

// AddBias broadcasts a column vector across every column of x, the way a
// bias joins a batch of activations. Because the one bias contributed to
// every column, its gradient is the sum of the incoming gradient along the
// batch -- broadcasting forward is summing backward.
func AddBias(x, bias *Node) *Node {
	if bias.Value.Columns != 1 || bias.Value.Rows != x.Value.Rows {
		panic("autograd: AddBias needs a column vector matching x's rows")
	}

	return newNode(x.Value.AddColumn(bias.Value), func(grad matrix.Matrix) {
		push(x, grad)
		push(bias, grad.RowSums())
	}, x, bias)
}

// Scale multiplies by a constant.
func Scale(a *Node, k float64) *Node {
	return newNode(a.Value.Scale(k), func(grad matrix.Matrix) {
		push(a, grad.Scale(k))
	}, a)
}

// Transpose swaps rows and columns; the gradient transposes back.
func Transpose(a *Node) *Node {
	return newNode(a.Value.Transpose(), func(grad matrix.Matrix) {
		push(a, grad.Transpose())
	}, a)
}

// Sum reduces every element to a single scalar node. Since each element
// contributed exactly once, the incoming scalar gradient is copied to all
// of them.
func Sum(a *Node) *Node {
	value := matrix.New(1, 1, [][]float64{{a.Value.Sum()}})

	return newNode(value, func(grad matrix.Matrix) {
		g := grad.At(0, 0)
		push(a, a.Value.Map(func(_ float64, _, _ int) float64 { return g }))
	}, a)
}

// Mean reduces to the average of every element.
func Mean(a *Node) *Node {
	n := float64(a.Value.Rows * a.Value.Columns)
	value := matrix.New(1, 1, [][]float64{{a.Value.Sum() / n}})

	return newNode(value, func(grad matrix.Matrix) {
		g := grad.At(0, 0) / n
		push(a, a.Value.Map(func(_ float64, _, _ int) float64 { return g }))
	}, a)
}

// Tanh applies the hyperbolic tangent elementwise. Its derivative, 1 - y^2,
// is read off the output -- which this engine has kept, so unlike the
// hand-written backprop in the parent package there is no requirement that
// an activation be recoverable from its own output.
func Tanh(a *Node) *Node {
	value := a.Value.Map(func(v float64, _, _ int) float64 { return math.Tanh(v) })

	return newNode(value, func(grad matrix.Matrix) {
		push(a, grad.HadamardProduct(value.Map(func(y float64, _, _ int) float64 {
			return 1 - y*y
		})))
	}, a)
}

// Sigmoid applies the logistic function elementwise.
func Sigmoid(a *Node) *Node {
	value := a.Value.Map(func(v float64, _, _ int) float64 { return 1 / (1 + math.Exp(-v)) })

	return newNode(value, func(grad matrix.Matrix) {
		push(a, grad.HadamardProduct(value.Map(func(y float64, _, _ int) float64 {
			return y * (1 - y)
		})))
	}, a)
}

// ReLU zeroes the negative half. The gradient is gated by the *input's*
// sign, which is the case the output-only convention elsewhere in this
// module cannot express for a non-invertible activation.
func ReLU(a *Node) *Node {
	value := a.Value.Map(func(v float64, _, _ int) float64 { return math.Max(0, v) })

	return newNode(value, func(grad matrix.Matrix) {
		push(a, grad.HadamardProduct(a.Value.Map(func(x float64, _, _ int) float64 {
			if x > 0 {
				return 1
			}
			return 0
		})))
	}, a)
}

// GELU is the Gaussian Error Linear Unit in its tanh approximation
// (Hendrycks and Gimpel, 2016): x weighted by roughly the probability that
// a standard normal falls below it, so small activations are damped
// smoothly rather than clipped. It is the activation transformers are
// built on, and it is a good example of what this engine buys -- GELU is
// not monotonic, so its derivative simply cannot be recovered from its
// output the way the parent package's Activation demands.
func GELU(a *Node) *Node {
	const c = 0.7978845608028654 // sqrt(2/pi)

	inner := a.Value.Map(func(x float64, _, _ int) float64 {
		return c * (x + 0.044715*x*x*x)
	})
	value := a.Value.Map(func(x float64, i, j int) float64 {
		return 0.5 * x * (1 + math.Tanh(inner.At(i, j)))
	})

	return newNode(value, func(grad matrix.Matrix) {
		push(a, grad.HadamardProduct(a.Value.Map(func(x float64, i, j int) float64 {
			t := math.Tanh(inner.At(i, j))
			// d/dx [0.5 x (1 + tanh(u))] with u = c(x + 0.044715 x^3).
			return 0.5*(1+t) + 0.5*x*(1-t*t)*c*(1+3*0.044715*x*x)
		})))
	}, a)
}

// Exp and Log are elementwise, and mostly exist so an expression can be
// written out by hand rather than reaching for a fused loss.
func Exp(a *Node) *Node {
	value := a.Value.Map(func(v float64, _, _ int) float64 { return math.Exp(v) })

	return newNode(value, func(grad matrix.Matrix) {
		push(a, grad.HadamardProduct(value))
	}, a)
}

// Log is the natural logarithm, undefined at or below zero.
func Log(a *Node) *Node {
	value := a.Value.Map(func(v float64, _, _ int) float64 { return math.Log(v) })

	return newNode(value, func(grad matrix.Matrix) {
		push(a, grad.HadamardProduct(a.Value.Map(func(x float64, _, _ int) float64 {
			return 1 / x
		})))
	}, a)
}

// Softmax normalizes each *column* into a probability distribution, the
// layout this package uses for a batch of samples (and, in attention, for
// one query's weights over all keys).
//
// Its backward pass is the one genuinely non-elementwise case here: every
// output in a column depends on every input in that column, so the
// gradient of an entry is y_i * (g_i - sum_j g_j y_j). The subtracted term
// is the column's own weighted average gradient, which is what enforces
// the constraint that a distribution's entries must sum to one -- pushing
// one probability up has to pull the others down.
func Softmax(a *Node) *Node {
	value := matrix.New(a.Value.Rows, a.Value.Columns, nil)
	for j := 0; j < a.Value.Columns; j++ {
		column := softmaxColumn(a.Value, j)
		for i, v := range column {
			value.Set(i, j, v)
		}
	}

	return newNode(value, func(grad matrix.Matrix) {
		out := matrix.New(value.Rows, value.Columns, nil)
		for j := 0; j < value.Columns; j++ {
			dot := 0.0
			for i := 0; i < value.Rows; i++ {
				dot += grad.At(i, j) * value.At(i, j)
			}
			for i := 0; i < value.Rows; i++ {
				out.Set(i, j, value.At(i, j)*(grad.At(i, j)-dot))
			}
		}
		push(a, out)
	}, a)
}

// SoftmaxCrossEntropy is softmax and categorical cross-entropy fused into
// one node: it takes raw logits and one-hot targets laid out one sample per
// column, and returns the mean loss over the batch.
//
// Fusing them is not just convenience. Composing Softmax with a separate
// log would exponentiate and then immediately take a logarithm, losing
// precision exactly where the network is most confident, and the two
// Jacobians would have to cancel numerically. Written together, the whole
// derivative collapses to (softmax - target) / batch -- the shortcut the
// hand-written optimizers in the parent package special-case by name, here
// falling out as just another backward closure.
func SoftmaxCrossEntropy(logits, targets *Node) *Node {
	checkSameShape(logits, targets, "SoftmaxCrossEntropy")

	batch := float64(logits.Value.Columns)
	probabilities := matrix.New(logits.Value.Rows, logits.Value.Columns, nil)

	loss := 0.0
	for j := 0; j < logits.Value.Columns; j++ {
		column := softmaxColumn(logits.Value, j)
		for i, p := range column {
			probabilities.Set(i, j, p)
			loss -= targets.Value.At(i, j) * math.Log(math.Max(p, 1e-12))
		}
	}

	value := matrix.New(1, 1, [][]float64{{loss / batch}})

	return newNode(value, func(grad matrix.Matrix) {
		g := grad.At(0, 0) / batch
		push(logits, probabilities.SubtractMatrix(targets.Value).Scale(g))

		// Soft targets are differentiable as well, which is what makes
		// label smoothing or a teacher's distribution expressible here:
		// d(loss)/d(target) is just the negative log probability.
		push(targets, probabilities.Map(func(p float64, _, _ int) float64 {
			return -math.Log(math.Max(p, 1e-12)) * g
		}))
	}, logits, targets)
}

// MSELoss is the mean squared error over every element of the batch.
//
// The target is differentiated too, rather than assumed constant. Wrapping
// a fixed label in Const already stops the gradient at no cost, and
// leaving the derivative in place is what lets a target itself be
// computed -- one network's output used as another's objective, as in
// knowledge distillation.
func MSELoss(prediction, target *Node) *Node {
	checkSameShape(prediction, target, "MSELoss")

	residual := prediction.Value.SubtractMatrix(target.Value)
	n := float64(residual.Rows * residual.Columns)
	value := matrix.New(1, 1, [][]float64{{residual.HadamardProduct(residual).Sum() / n}})

	return newNode(value, func(grad matrix.Matrix) {
		step := residual.Scale(2 * grad.At(0, 0) / n)
		push(prediction, step)
		push(target, step.Scale(-1))
	}, prediction, target)
}

// softmaxColumn returns one column of a matrix as a probability
// distribution, shifted by the column max before exponentiating so large
// logits cannot overflow (softmax is invariant to that shift).
func softmaxColumn(m matrix.Matrix, j int) []float64 {
	max := math.Inf(-1)
	for i := 0; i < m.Rows; i++ {
		max = math.Max(max, m.At(i, j))
	}

	out := make([]float64, m.Rows)
	sum := 0.0
	for i := 0; i < m.Rows; i++ {
		out[i] = math.Exp(m.At(i, j) - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}

	return out
}

func checkSameShape(a, b *Node, op string) {
	if a.Value.Rows != b.Value.Rows || a.Value.Columns != b.Value.Columns {
		panic("autograd: " + op + " needs operands of the same shape")
	}
}
