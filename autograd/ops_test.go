package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

// checkGradients is the engine's proof of correctness: it compares every
// analytic gradient against a central finite difference of the forward
// pass. build is called with fresh parameter nodes each time, since a
// graph holds the values it was built from and a perturbed evaluation
// needs a new one.
//
// A finite difference knows nothing about the chain rule -- it just nudges
// an input and watches the output -- so agreement means the backward
// closures really do implement the derivative of the forward code.
func checkGradients(t *testing.T, name string, build func(params []*Node) *Node, values ...matrix.Matrix) {
	t.Helper()

	const (
		h   = 1e-6
		tol = 1e-6
	)

	forward := func(perturbed []matrix.Matrix) float64 {
		nodes := make([]*Node, len(perturbed))
		for i, v := range perturbed {
			nodes[i] = Param(v)
		}
		return build(nodes).Item()
	}

	nodes := make([]*Node, len(values))
	for i, v := range values {
		nodes[i] = Param(v.Copy())
	}

	loss := build(nodes)
	loss.Backward()

	for p := range values {
		for i := 0; i < values[p].Rows; i++ {
			for j := 0; j < values[p].Columns; j++ {
				bumped := func(offset float64) float64 {
					copies := make([]matrix.Matrix, len(values))
					for k, v := range values {
						copies[k] = v.Copy()
					}
					copies[p].Set(i, j, copies[p].At(i, j)+offset)
					return forward(copies)
				}

				numeric := (bumped(h) - bumped(-h)) / (2 * h)
				analytic := nodes[p].Grad.At(i, j)

				if math.Abs(numeric-analytic) > tol*math.Max(1, math.Abs(numeric)) {
					t.Errorf("%s: param %d grad[%d][%d] = %g, want the numeric %g",
						name, p, i, j, analytic, numeric)
				}
			}
		}
	}
}

func randomMatrix(r, c int) matrix.Matrix {
	m := matrix.New(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, rand.NormFloat64())
		}
	}
	return m
}

// TestEveryOperationMatchesNumericGradient walks each operation through the
// finite-difference check, wrapped in a reduction so the graph ends in a
// scalar.
func TestEveryOperationMatchesNumericGradient(t *testing.T) {
	rand.Seed(7)

	for _, tc := range []struct {
		name   string
		build  func(p []*Node) *Node
		shapes [][2]int
	}{
		{"add", func(p []*Node) *Node { return Sum(Add(p[0], p[1])) }, [][2]int{{3, 2}, {3, 2}}},
		{"sub", func(p []*Node) *Node { return Sum(Sub(p[0], p[1])) }, [][2]int{{3, 2}, {3, 2}}},
		{"mul", func(p []*Node) *Node { return Sum(Mul(p[0], p[1])) }, [][2]int{{3, 2}, {3, 2}}},
		{"matmul", func(p []*Node) *Node { return Sum(MatMul(p[0], p[1])) }, [][2]int{{3, 4}, {4, 2}}},
		{"addbias", func(p []*Node) *Node { return Sum(AddBias(p[0], p[1])) }, [][2]int{{3, 5}, {3, 1}}},
		{"scale", func(p []*Node) *Node { return Sum(Scale(p[0], -2.5)) }, [][2]int{{3, 2}}},
		{"transpose", func(p []*Node) *Node { return Sum(Transpose(p[0])) }, [][2]int{{3, 2}}},
		{"mean", func(p []*Node) *Node { return Mean(p[0]) }, [][2]int{{3, 4}}},
		{"tanh", func(p []*Node) *Node { return Sum(Tanh(p[0])) }, [][2]int{{3, 2}}},
		{"sigmoid", func(p []*Node) *Node { return Sum(Sigmoid(p[0])) }, [][2]int{{3, 2}}},
		{"gelu", func(p []*Node) *Node { return Sum(GELU(p[0])) }, [][2]int{{3, 2}}},
		{"exp", func(p []*Node) *Node { return Sum(Exp(p[0])) }, [][2]int{{3, 2}}},
		{"softmax", func(p []*Node) *Node { return Sum(Mul(Softmax(p[0]), p[1])) }, [][2]int{{4, 3}, {4, 3}}},
		{"mse", func(p []*Node) *Node { return MSELoss(p[0], p[1]) }, [][2]int{{3, 2}, {3, 2}}},

		// Both sides of each loss differentiated, which is what makes a
		// computed target (distillation, soft labels) expressible.
		{"soft_cross_entropy", func(p []*Node) *Node {
			return SoftmaxCrossEntropy(p[0], Softmax(p[1]))
		}, [][2]int{{4, 3}, {4, 3}}},

		// A composite: two dense layers, a non-elementwise activation and a
		// reduction, which is where a wrong transpose or a missed
		// accumulation shows up even when each op passes alone.
		{"mlp", func(p []*Node) *Node {
			hidden := Tanh(AddBias(MatMul(p[0], p[1]), p[2]))
			return Mean(GELU(MatMul(p[3], hidden)))
		}, [][2]int{{4, 3}, {3, 5}, {4, 1}, {2, 4}}},
	} {
		values := make([]matrix.Matrix, len(tc.shapes))
		for i, shape := range tc.shapes {
			values[i] = randomMatrix(shape[0], shape[1])
		}

		checkGradients(t, tc.name, tc.build, values...)
	}
}

// TestLogGradient keeps its input positive, where the logarithm is defined.
func TestLogGradient(t *testing.T) {
	rand.Seed(7)

	values := randomMatrix(3, 2).Map(func(v float64, _, _ int) float64 {
		return math.Abs(v) + 0.5
	})

	checkGradients(t, "log", func(p []*Node) *Node { return Sum(Log(p[0])) }, values)
}

// TestReLUGradient probes away from the kink at zero, where a finite
// difference straddling the corner is meaningless.
func TestReLUGradient(t *testing.T) {
	values := matrix.New(2, 3, [][]float64{
		{1.5, -2.0, 0.7},
		{-0.4, 3.1, -1.2},
	})

	checkGradients(t, "relu", func(p []*Node) *Node { return Sum(ReLU(p[0])) }, values)
}

// TestSoftmaxCrossEntropyGradient checks the fused loss against the
// numeric derivative of its own forward pass, with the targets held
// constant the way training uses them.
func TestSoftmaxCrossEntropyGradient(t *testing.T) {
	rand.Seed(7)

	targets := matrix.New(3, 4, nil)
	for j := 0; j < 4; j++ {
		targets.Set(rand.Intn(3), j, 1)
	}

	checkGradients(t, "softmax_cross_entropy", func(p []*Node) *Node {
		return SoftmaxCrossEntropy(p[0], Const(targets))
	}, randomMatrix(3, 4))
}

// TestSoftmaxCrossEntropyMatchesTheShortcut pins the fused gradient
// against the closed form the parent package special-cases by name.
func TestSoftmaxCrossEntropyMatchesTheShortcut(t *testing.T) {
	rand.Seed(7)

	logits := Param(randomMatrix(4, 3))
	targets := matrix.New(4, 3, nil)
	for j := 0; j < 3; j++ {
		targets.Set(j%4, j, 1)
	}

	loss := SoftmaxCrossEntropy(logits, Const(targets))
	loss.Backward()

	probabilities := matrix.New(4, 3, nil)
	for j := 0; j < 3; j++ {
		for i, p := range softmaxColumn(logits.Value, j) {
			probabilities.Set(i, j, p)
		}
	}

	want := probabilities.SubtractMatrix(targets).Divide(3)
	if !logits.Grad.ApproxEqual(want, 1e-12) {
		t.Errorf("gradient is\n%v\nwant (softmax - target) / batch\n%v", logits.Grad, want)
	}
}

// TestReusedNodeAccumulatesBothPaths is the property that separates a real
// graph from a chain: a value consumed twice must collect gradient from
// both consumers. Here y = x * x, whose derivative is 2x only if the two
// occurrences each contribute.
func TestReusedNodeAccumulatesBothPaths(t *testing.T) {
	x := Param(matrix.New(1, 1, [][]float64{{3}}))

	loss := Sum(Mul(x, x))
	loss.Backward()

	if got, want := x.Grad.At(0, 0), 6.0; math.Abs(got-want) > 1e-12 {
		t.Errorf("d(x*x)/dx at 3 = %g, want %g", got, want)
	}
}

// TestDiamondGraphVisitsOnce checks the topological order handles a node
// feeding two branches that later rejoin -- the classic case where a naive
// walk either double-counts or propagates before a node's gradient is
// complete.
func TestDiamondGraphVisitsOnce(t *testing.T) {
	x := Param(matrix.New(1, 1, [][]float64{{2}}))

	left := Scale(x, 3)           // 3x
	right := Mul(x, x)            // x^2
	loss := Sum(Add(left, right)) // 3x + x^2, derivative 3 + 2x = 7

	loss.Backward()

	if got, want := x.Grad.At(0, 0), 7.0; math.Abs(got-want) > 1e-12 {
		t.Errorf("gradient through a diamond = %g, want %g", got, want)
	}
}

// TestZeroGradClearsTheGraph covers the accumulate-not-overwrite contract
// and the reset that a training loop depends on.
func TestZeroGradClearsTheGraph(t *testing.T) {
	x := Param(matrix.New(1, 1, [][]float64{{3}}))
	loss := Sum(Mul(x, x))

	loss.Backward()
	loss.Backward() // deliberately without a reset
	if got, want := x.Grad.At(0, 0), 12.0; math.Abs(got-want) > 1e-12 {
		t.Errorf("two passes accumulated %g, want %g", got, want)
	}

	loss.ZeroGrad()
	if got := x.Grad.At(0, 0); got != 0 {
		t.Errorf("after ZeroGrad the gradient is %g, want 0", got)
	}

	loss.Backward()
	if got, want := x.Grad.At(0, 0), 6.0; math.Abs(got-want) > 1e-12 {
		t.Errorf("after ZeroGrad one pass gave %g, want %g", got, want)
	}
}

// TestConstantsCollectNoGradient pins the Param/Const distinction.
func TestConstantsCollectNoGradient(t *testing.T) {
	weight := Param(matrix.New(1, 2, [][]float64{{1, 2}}))
	input := Const(matrix.New(2, 1, [][]float64{{3}, {4}}))

	loss := Sum(MatMul(weight, input))
	loss.Backward()

	if weight.Grad.At(0, 0) == 0 {
		t.Error("the parameter collected no gradient")
	}
	if input.Grad.Rows != 0 {
		t.Errorf("the constant collected a gradient: %v", input.Grad)
	}
	if input.RequiresGrad() {
		t.Error("Const reported RequiresGrad")
	}
}

func TestOperationsRejectBadShapes(t *testing.T) {
	a := Param(matrix.New(2, 3, nil))
	b := Param(matrix.New(3, 2, nil))

	for name, f := range map[string]func(){
		"add":        func() { Add(a, b) },
		"sub":        func() { Sub(a, b) },
		"mul":        func() { Mul(a, b) },
		"matmul":     func() { MatMul(a, Param(matrix.New(2, 2, nil))) },
		"addbias":    func() { AddBias(a, Param(matrix.New(3, 1, nil))) },
		"mse":        func() { MSELoss(a, b) },
		"backward":   func() { a.Backward() },
		"item":       func() { a.Item() },
		"crossentro": func() { SoftmaxCrossEntropy(a, b) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: expected a panic, got none", name)
				}
			}()
			f()
		}()
	}
}
