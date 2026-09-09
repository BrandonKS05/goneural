package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

// TestGRUGradientsThroughTime differentiates a cell unrolled over several
// steps, checking the two-multiply structure (gates first, then a candidate
// that reads the gated state) against finite differences of the whole
// chain.
func TestGRUGradientsThroughTime(t *testing.T) {
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
		cell := &GRUCell{
			Gates:     &Linear{Weight: p[0], Bias: p[1]},
			Candidate: &Linear{Weight: p[2], Bias: p[3]},
			hidden:    hidden,
		}

		_, state := cell.Run(inputs, cell.ZeroState(1))
		return MSELoss(state, Const(target))
	}

	checkGradients(t, "gru", build,
		randomMatrix(2*hidden, hidden+input),
		randomMatrix(2*hidden, 1),
		randomMatrix(hidden, hidden+input),
		randomMatrix(hidden, 1))
}

// TestGRUUpdateGateStartsOpen pins the transplanted forget-gate trick: the
// update slice of the gate bias starts at 1, everything else at 0.
func TestGRUUpdateGateStartsOpen(t *testing.T) {
	const hidden = 3
	cell := NewGRUCell(2, hidden)

	for i := 0; i < hidden; i++ {
		if got := cell.Gates.Bias.Value.At(i, 0); got != 1 {
			t.Errorf("update bias[%d] = %g, want 1", i, got)
		}
	}
	for i := hidden; i < 2*hidden; i++ {
		if got := cell.Gates.Bias.Value.At(i, 0); got != 0 {
			t.Errorf("bias[%d] = %g, want 0 outside the update slice", i, got)
		}
	}
	for i := 0; i < hidden; i++ {
		if got := cell.Candidate.Bias.Value.At(i, 0); got != 0 {
			t.Errorf("candidate bias[%d] = %g, want 0", i, got)
		}
	}
}

// TestGRUCarriesStateAcrossManySteps checks the convex blend really can
// preserve information: with the update gate forced open the state must
// pass through unchanged, whatever the candidate proposes.
func TestGRUCarriesStateAcrossManySteps(t *testing.T) {
	rand.Seed(7)

	const (
		hidden = 2
		steps  = 50
	)

	cell := NewGRUCell(1, hidden)

	// Force the update gate open; the candidate weights stay random, which
	// is the point -- a wide-open gate must ignore whatever they propose.
	for i := 0; i < hidden; i++ {
		cell.Gates.Bias.Value.Set(i, 0, 20)
	}
	cell.Gates.Weight.Value = cell.Gates.Weight.Value.Scale(0)

	state := Const(matrix.New(hidden, 1, [][]float64{{0.75}, {-0.4}}))
	for step := 0; step < steps; step++ {
		state = cell.Step(Const(matrix.New(1, 1, nil)), state)
	}

	for i, want := range []float64{0.75, -0.4} {
		if got := state.Value.At(i, 0); math.Abs(got-want) > 1e-6 {
			t.Errorf("after %d steps state[%d] = %g, want the original %g", steps, i, got, want)
		}
	}
}

// TestGRUGradientSurvivesLongSequences is the same claim measured on the
// backward pass: with the update gate mostly open, gradient from the last
// step must still reach the first input rather than vanishing.
func TestGRUGradientSurvivesLongSequences(t *testing.T) {
	rand.Seed(7)

	const (
		hidden = 4
		steps  = 60
	)

	cell := NewGRUCell(2, hidden)
	for i := 0; i < hidden; i++ {
		cell.Gates.Bias.Value.Set(i, 0, 3) // a firmly open update gate
	}

	first := Param(randomMatrix(2, 1))
	inputs := []*Node{first}
	for i := 1; i < steps; i++ {
		inputs = append(inputs, Const(randomMatrix(2, 1)))
	}

	_, state := cell.Run(inputs, cell.ZeroState(1))
	Sum(state).Backward()

	norm := 0.0
	for _, g := range first.Grad.Flatten() {
		norm += g * g
	}

	if norm = math.Sqrt(norm); norm < 1e-6 {
		t.Errorf("gradient reaching the first of %d steps has norm %g, want it to survive", steps, norm)
	}
}

// TestGRULearnsToRemember trains the cell on the same memory task the LSTM
// faces: the label is decided by the first token of a sequence and only
// asked for at the very end, so two gates must do the work of three.
func TestGRULearnsToRemember(t *testing.T) {
	rand.Seed(11)

	const (
		hidden = 12
		length = 12
		steps  = 500
	)

	cell := NewGRUCell(3, hidden)
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
		return head.Forward(state)
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

func TestGRURejectsBadShapes(t *testing.T) {
	cell := NewGRUCell(2, 3)

	for name, f := range map[string]func(){
		"zero input":  func() { NewGRUCell(0, 3) },
		"zero hidden": func() { NewGRUCell(2, 0) },
		"zero batch":  func() { cell.ZeroState(0) },
		"batch clash": func() { cell.Step(Const(matrix.New(2, 4, nil)), cell.ZeroState(1)) },
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
