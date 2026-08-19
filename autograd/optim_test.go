package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

// xorBatch returns the four XOR samples as one batch: inputs two rows by
// four columns (features down, samples across) and one-hot targets.
func xorBatch() (inputs, targets matrix.Matrix) {
	inputs = matrix.New(2, 4, [][]float64{
		{0, 0, 1, 1},
		{0, 1, 0, 1},
	})
	targets = matrix.New(2, 4, [][]float64{
		{1, 0, 0, 1}, // class 0: XOR is false
		{0, 1, 1, 0}, // class 1: XOR is true
	})
	return inputs, targets
}

// trainXOR builds a two-layer network out of graph operations and trains
// it with the given optimizer, returning the first and last loss.
func trainXOR(t *testing.T, optimizer Optimizer, steps int) (first, last float64, predict func() matrix.Matrix) {
	t.Helper()

	inputs, targets := xorBatch()

	scale := func(m matrix.Matrix, k float64) matrix.Matrix {
		return m.Map(func(_ float64, _, _ int) float64 { return rand.NormFloat64() * k })
	}

	w1 := Param(scale(matrix.New(8, 2, nil), 1)).Named("w1")
	b1 := Param(matrix.New(8, 1, nil)).Named("b1")
	w2 := Param(scale(matrix.New(2, 8, nil), 0.5)).Named("w2")
	b2 := Param(matrix.New(2, 1, nil)).Named("b2")

	params := []*Node{w1, b1, w2, b2}

	forward := func() *Node {
		hidden := Tanh(AddBias(MatMul(w1, Const(inputs)), b1))
		return AddBias(MatMul(w2, hidden), b2)
	}

	for step := 0; step < steps; step++ {
		loss := SoftmaxCrossEntropy(forward(), Const(targets))

		ZeroGrad(params...)
		loss.Backward()
		optimizer.Step(params...)

		if step == 0 {
			first = loss.Item()
		}
		last = loss.Item()
	}

	return first, last, func() matrix.Matrix { return forward().Value }
}

// TestAdamLearnsXOR is the end-to-end proof that the engine trains: a
// network written purely as graph operations, with no hand-derived
// backprop anywhere, has to solve the problem the whole module is built
// around.
func TestAdamLearnsXOR(t *testing.T) {
	rand.Seed(7)

	first, last, predict := trainXOR(t, Adam(0.1), 400)
	if last >= first {
		t.Fatalf("loss went from %g to %g, want a decrease", first, last)
	}

	logits := predict()
	_, targets := xorBatch()
	for j := 0; j < logits.Columns; j++ {
		predicted := 0
		if logits.At(1, j) > logits.At(0, j) {
			predicted = 1
		}
		want := 0
		if targets.At(1, j) == 1 {
			want = 1
		}
		if predicted != want {
			t.Errorf("sample %d classified as %d, want %d", j, predicted, want)
		}
	}
	if last > 0.05 {
		t.Errorf("final loss %g, want it near zero", last)
	}
}

func TestSGDWithMomentumLearnsXOR(t *testing.T) {
	rand.Seed(7)

	first, last, _ := trainXOR(t, SGD(0.5, 0.9), 800)
	if last >= first {
		t.Fatalf("loss went from %g to %g, want a decrease", first, last)
	}
	if last > 0.1 {
		t.Errorf("final loss %g, want it near zero", last)
	}
}

// TestSGDStepsDownTheGradient checks one update in isolation against the
// arithmetic, including the momentum recurrence on the second step.
func TestSGDStepsDownTheGradient(t *testing.T) {
	x := Param(matrix.New(1, 1, [][]float64{{3}}))
	o := SGD(0.1, 0.5)

	// d(x^2)/dx = 2x = 6, so the first step is 3 - 0.1*6 = 2.4.
	loss := Sum(Mul(x, x))
	loss.Backward()
	o.Step(x)
	if got, want := x.Value.At(0, 0), 2.4; math.Abs(got-want) > 1e-12 {
		t.Fatalf("after one step x = %g, want %g", got, want)
	}

	// Second step: gradient 4.8, velocity 0.5*0.6 + 0.48 = 0.78.
	ZeroGrad(x)
	loss = Sum(Mul(x, x))
	loss.Backward()
	o.Step(x)
	if got, want := x.Value.At(0, 0), 2.4-0.78; math.Abs(got-want) > 1e-12 {
		t.Errorf("after two steps x = %g, want %g", got, want)
	}
}

// TestAdamNormalizesStepSize pins Adam's defining behaviour: the first
// step is the learning rate regardless of how large or small the gradient
// is, because the step divides the gradient by its own running magnitude.
func TestAdamNormalizesStepSize(t *testing.T) {
	for _, gradient := range []float64{1e-4, 1, 1e4} {
		x := Param(matrix.New(1, 1, [][]float64{{0}}))
		x.Grad = matrix.New(1, 1, [][]float64{{gradient}})

		Adam(0.01).Step(x)

		if got, want := x.Value.At(0, 0), -0.01; math.Abs(got-want) > 1e-6 {
			t.Errorf("gradient %g moved the parameter to %g, want %g", gradient, got, want)
		}
	}
}

// TestGradientAccumulationSumsBatches covers the documented leaf
// behaviour: several backward passes before a step add up, which is how a
// batch too large to hold at once is trained.
func TestGradientAccumulationSumsBatches(t *testing.T) {
	x := Param(matrix.New(1, 1, [][]float64{{3}}))

	ZeroGrad(x)
	for i := 0; i < 3; i++ {
		Sum(Mul(x, x)).Backward()
	}

	if got, want := x.Grad.At(0, 0), 18.0; math.Abs(got-want) > 1e-12 {
		t.Errorf("three accumulated passes gave %g, want %g", got, want)
	}
}

func TestOptimizersRejectBadParameters(t *testing.T) {
	for name, f := range map[string]func(){
		"zero lr":         func() { SGD(0, 0.9) },
		"negative lr":     func() { Adam(-1) },
		"momentum of one": func() { SGD(0.1, 1) },
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
