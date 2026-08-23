package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

// TestLSTMGradientsThroughTime differentiates a cell unrolled over several
// steps. Nothing implements backpropagation through time -- the unrolled
// graph is just a long chain -- so this checks the general engine really
// does handle it.
func TestLSTMGradientsThroughTime(t *testing.T) {
	rand.Seed(7)

	const (
		input  = 3
		hidden = 4
		steps  = 5
	)

	inputs := make([]*Node, steps)
	for i := range inputs {
		inputs[i] = Const(randomMatrix(input, 1))
	}
	target := randomMatrix(hidden, 1)

	build := func(p []*Node) *Node {
		cell := &LSTMCell{
			Gates:  &Linear{Weight: p[0], Bias: p[1]},
			hidden: hidden,
		}

		_, state := cell.Run(inputs, cell.ZeroState(1))
		return MSELoss(state.Hidden, Const(target))
	}

	checkGradients(t, "lstm", build,
		randomMatrix(4*hidden, hidden+input),
		randomMatrix(4*hidden, 1))
}

// TestLSTMForgetGateStartsOpen pins the initialization detail that decides
// whether the cell can remember anything at all early in training.
func TestLSTMForgetGateStartsOpen(t *testing.T) {
	const hidden = 3
	cell := NewLSTMCell(2, hidden)

	for i := 0; i < hidden; i++ {
		if got := cell.Gates.Bias.Value.At(i, 0); got != 1 {
			t.Errorf("forget bias[%d] = %g, want 1", i, got)
		}
	}
	for i := hidden; i < 4*hidden; i++ {
		if got := cell.Gates.Bias.Value.At(i, 0); got != 0 {
			t.Errorf("bias[%d] = %g, want 0 outside the forget slice", i, got)
		}
	}
}

// TestLSTMCarriesStateAcrossManySteps is the property the architecture
// exists for: information from the first step must still be reachable at
// the end of a long sequence. With the forget gates forced open and the
// input gates shut, the cell should hold its value unchanged.
func TestLSTMCarriesStateAcrossManySteps(t *testing.T) {
	const (
		hidden = 2
		steps  = 50
	)

	cell := NewLSTMCell(1, hidden)

	// Force forget open (large positive bias) and input shut (large
	// negative), so the cell state is meant to be preserved exactly.
	for i := 0; i < hidden; i++ {
		cell.Gates.Bias.Value.Set(i, 0, 20)         // forget
		cell.Gates.Bias.Value.Set(hidden+i, 0, -20) // input
	}
	cell.Gates.Weight.Value = cell.Gates.Weight.Value.Scale(0)

	state := cell.ZeroState(1)
	state.Cell = Const(matrix.New(hidden, 1, [][]float64{{0.75}, {-0.4}}))

	for step := 0; step < steps; step++ {
		state = cell.Step(Const(matrix.New(1, 1, nil)), state)
	}

	for i, want := range []float64{0.75, -0.4} {
		if got := state.Cell.Value.At(i, 0); math.Abs(got-want) > 1e-6 {
			t.Errorf("after %d steps cell[%d] = %g, want the original %g", steps, i, got, want)
		}
	}
}

// TestLSTMGradientSurvivesLongSequences is the same claim measured on the
// backward pass: with the memory path open, gradient from the last step
// must still reach the first input, rather than vanishing.
func TestLSTMGradientSurvivesLongSequences(t *testing.T) {
	rand.Seed(7)

	const (
		hidden = 4
		steps  = 60
	)

	cell := NewLSTMCell(2, hidden)
	for i := 0; i < hidden; i++ {
		cell.Gates.Bias.Value.Set(i, 0, 3) // a firmly open forget gate
	}

	first := Param(randomMatrix(2, 1))
	inputs := []*Node{first}
	for i := 1; i < steps; i++ {
		inputs = append(inputs, Const(randomMatrix(2, 1)))
	}

	_, state := cell.Run(inputs, cell.ZeroState(1))
	Sum(state.Hidden).Backward()

	norm := 0.0
	for _, g := range first.Grad.Flatten() {
		norm += g * g
	}

	if norm = math.Sqrt(norm); norm < 1e-6 {
		t.Errorf("gradient reaching the first of %d steps has norm %g, want it to survive", steps, norm)
	}
}

// TestLSTMLearnsToRemember trains the cell on a task that cannot be solved
// without memory: the label is decided by the *first* token of a sequence,
// and only revealed by a prediction made at the very end. A model with no
// state can only guess.
func TestLSTMLearnsToRemember(t *testing.T) {
	rand.Seed(11)

	const (
		hidden = 12
		length = 12
		steps  = 500
	)

	cell := NewLSTMCell(3, hidden)
	head := NewLinear(hidden, 2)
	params := append(cell.Parameters(), head.Parameters()...)
	optimizer := Adam(0.03)

	// Each sequence starts with a one-hot signal in row 0 or row 1, then
	// carries only noise in row 2.
	sample := func() ([]*Node, int) {
		class := rand.Intn(2)

		inputs := make([]*Node, length)
		for i := range inputs {
			step := matrix.New(3, 1, nil)
			if i == 0 {
				step.Set(class, 0, 1)
			} else {
				step.Set(2, 0, rand.NormFloat64()*0.5)
			}
			inputs[i] = Const(step)
		}

		return inputs, class
	}

	forward := func(inputs []*Node) *Node {
		_, state := cell.Run(inputs, cell.ZeroState(1))
		return head.Forward(state.Hidden)
	}

	for step := 0; step < steps; step++ {
		inputs, class := sample()

		target := matrix.New(2, 1, nil)
		target.Set(class, 0, 1)

		loss := SoftmaxCrossEntropy(forward(inputs), Const(target))

		ZeroGrad(params...)
		loss.Backward()
		ClipGradients(5, params...)
		optimizer.Step(params...)
	}

	correct, trials := 0, 150
	for i := 0; i < trials; i++ {
		inputs, class := sample()

		logits := forward(inputs).Value
		predicted := 0
		if logits.At(1, 0) > logits.At(0, 0) {
			predicted = 1
		}
		if predicted == class {
			correct++
		}
	}

	if accuracy := float64(correct) / float64(trials); accuracy < 0.95 {
		t.Errorf("recall accuracy %.3f after %d tokens of noise, want the cell to remember", accuracy, length-1)
	}
}

// TestClipGradientsRescalesTheWholeSet checks the norm is measured across
// every parameter at once and that direction is preserved.
func TestClipGradientsRescalesTheWholeSet(t *testing.T) {
	a := Param(matrix.New(1, 2, nil))
	b := Param(matrix.New(1, 1, nil))

	a.Grad = matrix.New(1, 2, [][]float64{{3, 0}})
	b.Grad = matrix.New(1, 1, [][]float64{{4}}) // combined norm is 5

	if got, want := ClipGradients(2.5, a, b), 5.0; math.Abs(got-want) > 1e-12 {
		t.Errorf("reported norm %g, want %g", got, want)
	}

	// Halved, since the limit is half the norm -- and the ratio between
	// the two gradients is unchanged.
	if got := a.Grad.At(0, 0); math.Abs(got-1.5) > 1e-12 {
		t.Errorf("a clipped to %g, want 1.5", got)
	}
	if got := b.Grad.At(0, 0); math.Abs(got-2) > 1e-12 {
		t.Errorf("b clipped to %g, want 2", got)
	}
}

// TestClipGradientsLeavesSmallOnesAlone pins the no-op case.
func TestClipGradientsLeavesSmallOnesAlone(t *testing.T) {
	p := Param(matrix.New(1, 1, nil))
	p.Grad = matrix.New(1, 1, [][]float64{{0.5}})

	if got, want := ClipGradients(10, p), 0.5; math.Abs(got-want) > 1e-12 {
		t.Errorf("reported norm %g, want %g", got, want)
	}
	if got := p.Grad.At(0, 0); got != 0.5 {
		t.Errorf("gradient became %g, want it untouched", got)
	}

	// An all-zero gradient must not divide by zero.
	ZeroGrad(p)
	if got := ClipGradients(1, p); got != 0 {
		t.Errorf("zero gradients reported norm %g, want 0", got)
	}
}

func TestRowsSlicesAndRestores(t *testing.T) {
	x := Param(matrix.New(4, 2, [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
		{7, 8},
	}))

	slice := Rows(x, 1, 2)
	want := matrix.New(2, 2, [][]float64{{3, 4}, {5, 6}})
	if !slice.Value.Equal(want) {
		t.Fatalf("Rows gave\n%v\nwant\n%v", slice.Value, want)
	}

	Sum(slice).Backward()
	for i := 0; i < 4; i++ {
		for j := 0; j < 2; j++ {
			want := 0.0
			if i == 1 || i == 2 {
				want = 1
			}
			if got := x.Grad.At(i, j); got != want {
				t.Errorf("grad (%d, %d) = %g, want %g", i, j, got, want)
			}
		}
	}
}

func TestRecurrentRejectsBadShapes(t *testing.T) {
	cell := NewLSTMCell(2, 3)

	for name, f := range map[string]func(){
		"zero input":  func() { NewLSTMCell(0, 3) },
		"zero hidden": func() { NewLSTMCell(2, 0) },
		"zero batch":  func() { cell.ZeroState(0) },
		"batch clash": func() { cell.Step(Const(matrix.New(2, 4, nil)), cell.ZeroState(1)) },
		"bad slice":   func() { Rows(Const(matrix.New(3, 1, nil)), 2, 5) },
		"bad clip":    func() { ClipGradients(0, cell.Parameters()...) },
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
