package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

func TestGatherAndScatterGradients(t *testing.T) {
	rand.Seed(7)

	// Index 1 appears twice, so its gradient has to accumulate.
	checkGradients(t, "gather", func(p []*Node) *Node {
		return Sum(Mul(GatherColumns(p[0], []int{2, 1, 1}), p[1]))
	}, randomMatrix(3, 4), randomMatrix(3, 3))

	checkGradients(t, "scatter", func(p []*Node) *Node {
		return Sum(Mul(ScatterColumns(p[0], []int{3, 0, 0}, 4), p[1]))
	}, randomMatrix(2, 3), randomMatrix(2, 4))

	checkGradients(t, "div", func(p []*Node) *Node {
		// Denominator kept away from zero.
		return Sum(Div(p[0], Add(p[1], Const(matrix.New(2, 3, nil).
			Map(func(_ float64, _, _ int) float64 { return 5 })))))
	}, randomMatrix(2, 3), randomMatrix(2, 3))

	checkGradients(t, "sum2", func(p []*Node) *Node {
		return Sum(Mul(Sum2(p[0]), p[1]))
	}, randomMatrix(3, 4), randomMatrix(3, 1))
}

// TestGatherScatterRoundTrip checks the two are inverses on a permutation,
// which is the case routing relies on.
func TestGatherScatterRoundTrip(t *testing.T) {
	rand.Seed(7)

	x := Const(randomMatrix(3, 4))
	order := []int{2, 0, 3, 1}

	back := ScatterColumns(GatherColumns(x, order), order, 4)
	if !back.Value.ApproxEqual(x.Value, 1e-12) {
		t.Errorf("gather then scatter gave\n%v\nwant\n%v", back.Value, x.Value)
	}
}

// TestMixtureRoutesEveryPosition checks the gate's contract: each position
// goes to exactly Active experts, its weights sum to one, and the reported
// load adds up.
func TestMixtureRoutesEveryPosition(t *testing.T) {
	rand.Seed(7)

	const (
		model     = 6
		experts   = 5
		active    = 2
		positions = 9
	)

	m := NewMixtureOfExperts(model, 8, experts, active)
	routing := m.Route(Const(randomMatrix(model, positions)))

	for j := 0; j < positions; j++ {
		if got := len(routing.Experts[j]); got != active {
			t.Errorf("position %d routed to %d experts, want %d", j, got, active)
		}

		seen := map[int]bool{}
		sum := 0.0
		for i, e := range routing.Experts[j] {
			if seen[e] {
				t.Errorf("position %d routed to expert %d twice", j, e)
			}
			seen[e] = true
			sum += routing.Weights[j][i]
		}
		if math.Abs(sum-1) > 1e-12 {
			t.Errorf("position %d weights sum to %g, want 1", j, sum)
		}
	}

	total := 0.0
	for _, fraction := range routing.Load {
		total += fraction
	}
	if want := float64(active); math.Abs(total-want) > 1e-9 {
		t.Errorf("load sums to %g, want %g", total, want)
	}
}

// TestMixtureGradients differentiates the whole layer, router included.
// Finite differences are valid here only as long as a perturbation does
// not flip a routing decision, so the check runs at a point where the gate
// is not near a tie.
func TestMixtureGradients(t *testing.T) {
	rand.Seed(3)

	const (
		model     = 4
		hidden    = 5
		experts   = 3
		active    = 2
		positions = 4
	)

	m := NewMixtureOfExperts(model, hidden, experts, active)
	input := randomMatrix(model, positions)
	target := randomMatrix(model, positions)

	params := m.Parameters()
	values := make([]matrix.Matrix, len(params))
	for i, p := range params {
		values[i] = p.Value.Copy()
	}

	build := func(p []*Node) *Node {
		rebuilt := &MixtureOfExperts{
			Router: &Linear{Weight: p[0], Bias: p[1]},
			Active: active,
		}
		for e := 0; e < experts; e++ {
			base := 2 + 4*e
			rebuilt.Up = append(rebuilt.Up, &Linear{Weight: p[base], Bias: p[base+1]})
			rebuilt.Down = append(rebuilt.Down, &Linear{Weight: p[base+2], Bias: p[base+3]})
		}

		out, balance := rebuilt.Forward(Const(input))
		return Add(MSELoss(out, Const(target)), Scale(balance, 0.01))
	}

	checkGradients(t, "mixture of experts", build, values...)
}

// TestRouterReceivesTaskGradient is the property that distinguishes a
// learned router from a fixed one: the task loss alone, with no balance
// term at all, must reach the router's weights.
func TestRouterReceivesTaskGradient(t *testing.T) {
	rand.Seed(7)

	m := NewMixtureOfExperts(4, 6, 3, 2)
	out, _ := m.Forward(Const(randomMatrix(4, 5)))

	MSELoss(out, Const(randomMatrix(4, 5))).Backward()

	norm := 0.0
	for _, g := range m.Router.Weight.Grad.Flatten() {
		norm += g * g
	}
	if math.Sqrt(norm) < 1e-9 {
		t.Error("the router received no gradient from the task loss")
	}
}

// TestBalanceLossPunishesCollapse checks the auxiliary term does its job:
// a router funnelling everything to one expert must score worse than one
// spreading traffic evenly.
func TestBalanceLossPunishesCollapse(t *testing.T) {
	rand.Seed(7)

	const (
		model     = 4
		experts   = 4
		positions = 16
	)

	balanced := NewMixtureOfExperts(model, 5, experts, 1)
	collapsed := NewMixtureOfExperts(model, 5, experts, 1)

	// Force the collapsed router to always pick expert 0, whatever it sees.
	collapsed.Router.Weight.Value = collapsed.Router.Weight.Value.Scale(0)
	collapsed.Router.Bias.Value = collapsed.Router.Bias.Value.Scale(0)
	collapsed.Router.Bias.Value.Set(0, 0, 10)

	// And the balanced one to spread evenly by position.
	balanced.Router.Weight.Value = balanced.Router.Weight.Value.Scale(0)
	balanced.Router.Bias.Value = balanced.Router.Bias.Value.Scale(0)
	for e := 0; e < experts; e++ {
		balanced.Router.Weight.Value.Set(e, e%model, 10)
	}

	input := matrix.New(model, positions, nil).Map(func(_ float64, i, j int) float64 {
		if i == j%model {
			return 1
		}
		return 0
	})

	_, evenly := balanced.Forward(Const(input))
	_, funnelled := collapsed.Forward(Const(input))

	if funnelled.Item() <= evenly.Item() {
		t.Errorf("collapsed router scored %g, want worse than the balanced %g",
			funnelled.Item(), evenly.Item())
	}

	// A perfectly uniform router scores 1 by construction; collapse scores
	// the expert count.
	// Not exactly the expert count: a softmax never quite reaches 1, so
	// the starved experts keep a sliver of probability.
	if want := float64(experts); math.Abs(funnelled.Item()-want) > 0.01 {
		t.Errorf("total collapse scored %g, want the expert count %g", funnelled.Item(), want)
	}
	if math.Abs(evenly.Item()-1) > 0.2 {
		t.Errorf("even routing scored %g, want about 1", evenly.Item())
	}
}

// TestUnusedExpertsCostNothing pins the sparsity claim: an expert nobody
// routed to must not appear in the graph, and so must receive no gradient.
func TestUnusedExpertsCostNothing(t *testing.T) {
	rand.Seed(7)

	m := NewMixtureOfExperts(4, 5, 3, 1)

	// Route everything to expert 0.
	m.Router.Weight.Value = m.Router.Weight.Value.Scale(0)
	m.Router.Bias.Value = m.Router.Bias.Value.Scale(0)
	m.Router.Bias.Value.Set(0, 0, 10)

	out, _ := m.Forward(Const(randomMatrix(4, 6)))
	Sum(out).Backward()

	if norm(m.Up[0].Weight.Grad) == 0 {
		t.Error("the chosen expert received no gradient")
	}
	for e := 1; e < m.Experts(); e++ {
		if got := norm(m.Up[e].Weight.Grad); got != 0 {
			t.Errorf("unrouted expert %d received gradient of norm %g", e, got)
		}
	}
}

func norm(m matrix.Matrix) float64 {
	total := 0.0
	for _, v := range m.Flatten() {
		total += v * v
	}
	return math.Sqrt(total)
}

// TestExpertsSpecialize trains the mixture on a task built from two
// unrelated sub-problems and checks the router learns to separate them --
// the behaviour the whole architecture is for. Inputs are tagged by kind
// in their first two rows, and the two kinds need opposite transforms.
func TestExpertsSpecialize(t *testing.T) {
	rand.Seed(5)

	// Two experts for two kinds of input: with more experts than kinds the
	// balance term would (correctly) split each kind across several, which
	// is fine for the task but makes "which expert owns this kind"
	// ambiguous to assert on.
	const (
		model   = 6
		experts = 2
		steps   = 900
	)

	m := NewMixtureOfExperts(model, 12, experts, 1)
	params := m.Parameters()
	optimizer := Adam(0.02)

	// Kind 0 must have its payload negated; kind 1 must have it doubled.
	// Samples come in batches, one per column: the balance term measures
	// how traffic spreads across a batch, so a batch of one carries no
	// balance signal at all and the router is free to collapse.
	const batch = 16

	sample := func() (matrix.Matrix, matrix.Matrix, []int) {
		in := matrix.New(model, batch, nil)
		want := matrix.New(model, batch, nil)
		kinds := make([]int, batch)

		for j := 0; j < batch; j++ {
			kind := rand.Intn(2)
			kinds[j] = kind
			in.Set(kind, j, 1)

			for i := 2; i < model; i++ {
				payload := rand.NormFloat64() * 0.5
				in.Set(i, j, payload)
				if kind == 0 {
					want.Set(i, j, -payload)
				} else {
					want.Set(i, j, 2*payload)
				}
			}
		}

		return in, want, kinds
	}

	for step := 0; step < steps; step++ {
		in, want, _ := sample()

		out, balance := m.Forward(Const(in))
		loss := Add(MSELoss(out, Const(want)), Scale(balance, 0.01))

		ZeroGrad(params...)
		loss.Backward()
		optimizer.Step(params...)
	}

	// Which expert handles which kind, and how consistently.
	counts := [2]map[int]int{{}, {}}
	errorTotal := 0.0

	const trials = 20
	for i := 0; i < trials; i++ {
		in, want, kinds := sample()

		routing := m.Route(Const(in))
		for j, kind := range kinds {
			counts[kind][routing.Experts[j][0]]++
		}

		out, _ := m.Forward(Const(in))
		errorTotal += MSELoss(out, Const(want)).Item()
	}

	if mean := errorTotal / trials; mean > 0.05 {
		t.Errorf("mean squared error %g, want the mixture to solve the task", mean)
	}

	dominant := func(counts map[int]int) (int, float64) {
		best, total := -1, 0
		for e, n := range counts {
			total += n
			if best == -1 || n > counts[best] {
				best = e
			}
		}
		return best, float64(counts[best]) / float64(total)
	}

	firstExpert, firstShare := dominant(counts[0])
	secondExpert, secondShare := dominant(counts[1])

	if firstShare < 0.9 || secondShare < 0.9 {
		t.Errorf("routing is inconsistent: kind 0 went to expert %d %.0f%% of the time, kind 1 to %d %.0f%%",
			firstExpert, firstShare*100, secondExpert, secondShare*100)
	}
	if firstExpert == secondExpert {
		t.Errorf("both kinds routed to expert %d, so the router did not specialize", firstExpert)
	}
}

func TestMixtureRejectsBadShapes(t *testing.T) {
	for name, f := range map[string]func(){
		"zero model":       func() { NewMixtureOfExperts(0, 4, 2, 1) },
		"zero experts":     func() { NewMixtureOfExperts(4, 4, 0, 1) },
		"too many active":  func() { NewMixtureOfExperts(4, 4, 2, 3) },
		"zero active":      func() { NewMixtureOfExperts(4, 4, 2, 0) },
		"empty gather":     func() { GatherColumns(Const(matrix.New(2, 2, nil)), nil) },
		"gather range":     func() { GatherColumns(Const(matrix.New(2, 2, nil)), []int{5}) },
		"scatter mismatch": func() { ScatterColumns(Const(matrix.New(2, 2, nil)), []int{0}, 4) },
		"scatter range":    func() { ScatterColumns(Const(matrix.New(2, 2, nil)), []int{0, 9}, 4) },
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
