package goneural

import "github.com/BrandonKS05/goneural/matrix"

// LookaheadOptimizer (Zhang et al., 2019) wraps any inner optimizer with a
// second, slower set of weights. The inner ("fast") optimizer explores as
// normal; every SyncPeriod invocations the slow weights take a step of size
// Alpha toward wherever the fast weights ended up, and the fast weights are
// reset onto the slow ones. Averaging over the fast optimizer's trajectory
// like this damps its oscillations and often buys extra stability
// essentially for free, since the inner optimizer needs no retuning.
type LookaheadOptimizer struct {
	Inner      Optimizer
	SyncPeriod int
	Alpha      float64

	slowW []matrix.Matrix
	slowB []matrix.Matrix
	step  int
}

// NewLookaheadOptimizer wraps inner with lookahead synchronization. The
// paper's defaults are a SyncPeriod of 5 and an Alpha of 0.5. Alpha must be
// in (0, 1]: 1 degrades to running the inner optimizer with periodic no-op
// resets, and anything outside that range stops being an interpolation.
func NewLookaheadOptimizer(inner Optimizer, syncPeriod int, alpha float64) *LookaheadOptimizer {
	if syncPeriod < 1 {
		panic("goneural: Lookahead needs a positive sync period")
	}
	if alpha <= 0 || alpha > 1 {
		panic("goneural: Lookahead alpha must be in (0, 1]")
	}

	return &LookaheadOptimizer{
		Inner:      inner,
		SyncPeriod: syncPeriod,
		Alpha:      alpha,
	}
}

// Lookahead returns an Optimizer that runs inner with lookahead slow
// weights (sync every k invocations, interpolating by alpha).
func Lookahead(inner Optimizer, syncPeriod int, alpha float64) Optimizer {
	return NewLookaheadOptimizer(inner, syncPeriod, alpha).Optimize
}

// Optimize implements the Optimizer signature.
func (o *LookaheadOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	if o.slowW == nil {
		// The slow weights start wherever the network starts.
		o.slowW = make([]matrix.Matrix, len(n.Weights))
		o.slowB = make([]matrix.Matrix, len(n.Biases))
		for l := range n.Weights {
			o.slowW[l] = n.Weights[l].Copy()
			o.slowB[l] = n.Biases[l].Copy()
		}
	}

	err := o.Inner(n, dataSet)

	o.step++
	if o.step%o.SyncPeriod == 0 {
		for l := range n.Weights {
			// slow += alpha * (fast - slow), then the fast weights restart
			// from the freshly moved slow ones.
			o.slowW[l] = o.slowW[l].AddMatrix(n.Weights[l].SubtractMatrix(o.slowW[l]).Scale(o.Alpha))
			o.slowB[l] = o.slowB[l].AddMatrix(n.Biases[l].SubtractMatrix(o.slowB[l]).Scale(o.Alpha))

			n.Weights[l] = o.slowW[l].Copy()
			n.Biases[l] = o.slowB[l].Copy()
		}
	}

	return err
}
