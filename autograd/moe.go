package autograd

import (
	"sort"

	"github.com/BrandonKS05/goneural/matrix"
)

// MixtureOfExperts replaces one feed-forward network with several, plus a
// small router that sends each position to only a few of them.
//
// The motivation is that a dense layer spends all of its parameters on
// every input. Capacity and cost are welded together: doubling what a
// model knows doubles what it costs to run. A mixture breaks that link.
// With sixteen experts and two active per position, the layer holds
// sixteen experts' worth of parameters while doing two experts' worth of
// arithmetic -- and because the router is learned, different experts end
// up specializing on different kinds of input rather than all learning the
// same average.
//
// The routing here is genuinely sparse, not masked: positions are grouped
// by which experts they chose, and each expert's network runs on its own
// columns only. That is what GatherColumns and ScatterColumns are for.
//
// The catch is load balance. Nothing in the loss says every expert must be
// used, and routers left alone collapse -- an expert that gets slightly
// more traffic early trains faster, attracts more traffic, and the rest
// starve. Forward returns an auxiliary loss for exactly this, which the
// caller adds to the training objective.
type MixtureOfExperts struct {
	// Router scores each expert for each position.
	Router *Linear

	// Up and Down are the per-expert feed-forward pair, one entry each.
	Up   []*Linear
	Down []*Linear

	// Active is how many experts each position is routed to. One is
	// cheapest; two is the common choice, since a position that is
	// borderline between two experts gets a blend rather than a coin flip,
	// which makes the router's decision boundary differentiable in
	// practice rather than a cliff.
	Active int
}

// NewMixtureOfExperts builds a mixture over the given model width with the
// given number of experts, each widening to hidden internally, and active
// experts consulted per position.
func NewMixtureOfExperts(model, hidden, experts, active int) *MixtureOfExperts {
	if model < 1 || hidden < 1 {
		panic("autograd: MixtureOfExperts needs positive dimensions")
	}
	if experts < 1 {
		panic("autograd: MixtureOfExperts needs at least one expert")
	}
	if active < 1 || active > experts {
		panic("autograd: MixtureOfExperts active count out of range")
	}

	m := &MixtureOfExperts{
		Router: NewLinear(model, experts),
		Active: active,
	}
	for e := 0; e < experts; e++ {
		m.Up = append(m.Up, NewLinear(model, hidden))
		m.Down = append(m.Down, NewLinear(hidden, model))
	}

	return m
}

// Experts reports how many experts the layer holds.
func (m *MixtureOfExperts) Experts() int {
	return len(m.Up)
}

// Parameters returns every learnable node: the router first, then each
// expert in turn.
func (m *MixtureOfExperts) Parameters() []*Node {
	params := m.Router.Parameters()
	for e := range m.Up {
		params = append(params, m.Up[e].Parameters()...)
		params = append(params, m.Down[e].Parameters()...)
	}
	return params
}

// Routing records which experts each position was sent to, for inspection
// and for the balance loss.
type Routing struct {
	// Experts[j] lists the expert ids position j was routed to, best
	// first, and Weights[j] the matching gate weights, renormalized to
	// sum to 1 across the chosen experts.
	Experts [][]int
	Weights [][]float64

	// Load[e] is the fraction of positions that chose expert e.
	Load []float64
}

// Route runs the gate and picks each position's experts without running
// any of them, which is what a diagnostic or a load report wants.
func (m *MixtureOfExperts) Route(x *Node) Routing {
	gates := Softmax(m.Router.Forward(x)).Value
	positions := gates.Columns

	routing := Routing{
		Experts: make([][]int, positions),
		Weights: make([][]float64, positions),
		Load:    make([]float64, m.Experts()),
	}

	for j := 0; j < positions; j++ {
		order := make([]int, m.Experts())
		for e := range order {
			order[e] = e
		}
		// Sorted by gate probability, ties going to the lower id so the
		// routing is reproducible.
		sort.SliceStable(order, func(a, b int) bool {
			return gates.At(order[a], j) > gates.At(order[b], j)
		})

		chosen := order[:m.Active]
		weights := make([]float64, m.Active)

		total := 0.0
		for i, e := range chosen {
			weights[i] = gates.At(e, j)
			total += weights[i]
		}
		for i := range weights {
			// Renormalized: the discarded experts' probability mass would
			// otherwise silently shrink the layer's output.
			weights[i] /= total
		}

		routing.Experts[j] = append([]int(nil), chosen...)
		routing.Weights[j] = weights
		for _, e := range chosen {
			routing.Load[e] += 1 / float64(positions)
		}
	}

	return routing
}

// Forward runs the mixture over an input of shape model x positions,
// returning the output and an auxiliary load-balancing loss.
//
// The auxiliary loss is the Switch Transformer's (Fedus et al., 2021):
// experts times the dot product of how much traffic each expert actually
// received with how much probability the router assigned it. It is
// minimized when both are uniform, and -- importantly -- it is
// differentiable through the router's probabilities even though the
// discrete routing decision is not, which is how a gradient reaches a
// collapsing router at all. Scale it small (0.01 is typical) before adding
// it to the real objective: it is a nudge toward balance, not the task.
func (m *MixtureOfExperts) Forward(x *Node) (output *Node, balance *Node) {
	positions := x.Value.Columns

	gateProbabilities := Softmax(m.Router.Forward(x))
	routing := m.Route(x)

	// Group positions by expert, so each expert's network runs once on all
	// of its traffic rather than once per position.
	assigned := make([][]int, m.Experts())
	for j := 0; j < positions; j++ {
		for _, e := range routing.Experts[j] {
			assigned[e] = append(assigned[e], j)
		}
	}

	// The renormalizing denominator, built as a node rather than a number:
	// each position's chosen gates, summed. Keeping it in the graph is
	// what lets the task loss -- not just the balance term -- tell the
	// router which expert was the better choice.
	//
	// With a single active expert there is nothing to renormalize against,
	// and doing it anyway would divide the one chosen gate by itself,
	// leaving a constant 1 and cutting the router off from the task loss
	// completely -- it could then only ever learn to balance, never to
	// route well. Top-1 therefore scales by the raw gate probability, as
	// the Switch Transformer does, and only top-k with k > 1 normalizes.
	var denominator *Node
	gathered := make([]*Node, m.Experts())
	for e := range m.Up {
		if len(assigned[e]) == 0 {
			continue
		}

		gathered[e] = GatherColumns(Rows(gateProbabilities, e, 1), assigned[e])
		placed := ScatterColumns(gathered[e], assigned[e], positions)

		if denominator == nil {
			denominator = placed
		} else {
			denominator = Add(denominator, placed)
		}
	}

	var parts []*Node
	for e := range m.Up {
		if len(assigned[e]) == 0 {
			continue // an expert nobody chose costs nothing at all
		}

		columns := GatherColumns(x, assigned[e])
		processed := m.Down[e].Forward(GELU(m.Up[e].Forward(columns)))

		// Each column scaled by its own gate weight, renormalized across
		// whichever experts that position chose.
		weight := gathered[e]
		if m.Active > 1 {
			weight = Div(weight, GatherColumns(denominator, assigned[e]))
		}

		scaled := Mul(processed, broadcastRow(weight, processed.Value.Rows))
		parts = append(parts, ScatterColumns(scaled, assigned[e], positions))
	}

	output = parts[0]
	for _, part := range parts[1:] {
		output = Add(output, part)
	}

	return output, m.balanceLoss(gateProbabilities, routing)
}

// balanceLoss builds the differentiable half of the load-balancing term:
// the measured load per expert is a constant (routing already happened),
// while the mean gate probability is not, so the gradient flows into the
// router.
func (m *MixtureOfExperts) balanceLoss(gates *Node, routing Routing) *Node {
	load := matrix.New(1, m.Experts(), nil)
	for e, fraction := range routing.Load {
		load.Set(0, e, fraction)
	}

	// Mean gate probability per expert: a row vector of the same shape.
	meanProbability := Scale(Transpose(Sum2(gates)), 1/float64(gates.Value.Columns))

	return Scale(Sum(Mul(Const(load), meanProbability)), float64(m.Experts()))
}

// Sum2 reduces a matrix along its columns, returning one row-sum per row
// as a column vector. It is the shape-preserving cousin of Sum, and exists
// because a per-expert or per-feature total is needed often enough that
// collapsing all the way to a scalar loses what you wanted.
func Sum2(x *Node) *Node {
	value := x.Value.RowSums()

	return newNode(value, func(grad matrix.Matrix) {
		// Each row's total was shared by every column in it, so the
		// gradient spreads back across the row unchanged.
		push(x, x.Value.Map(func(_ float64, i, _ int) float64 {
			return grad.At(i, 0)
		}))
	}, x)
}

// broadcastRow repeats a 1 x n row vector down to rows x n, so it can
// scale a whole column at a time.
func broadcastRow(row *Node, rows int) *Node {
	if row.Value.Rows != 1 {
		panic("autograd: broadcastRow needs a single-row node")
	}

	value := matrix.New(rows, row.Value.Columns, nil).
		Map(func(_ float64, _, j int) float64 { return row.Value.At(0, j) })

	return newNode(value, func(grad matrix.Matrix) {
		push(row, grad.ColumnSums())
	}, row)
}
