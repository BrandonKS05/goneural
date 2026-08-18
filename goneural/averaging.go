package goneural

import "github.com/BrandonKS05/goneural/matrix"

// WeightAveragerOptimizer wraps any inner optimizer and keeps a running
// average of the weights the optimizer visits. Training itself is
// untouched -- the inner optimizer sees the same network it always would,
// and the average lives off to the side until Apply copies it in.
//
// Averaging the tail of a training trajectory is one of the cheapest
// generalization wins available: stochastic optimizers do not converge to
// a point but bounce around a basin, and the centre of that bounce is
// usually flatter (and tests better) than any single point on the rim. Two
// classic variants share this machinery, differing only in how the running
// average is weighted:
//
//   - SWA, Stochastic Weight Averaging (Izmailov et al., 2018), an
//     equal-weight mean over every visited point. Every snapshot counts the
//     same no matter how old, so it converges to the true centre of the
//     explored region -- best paired with a cyclic or constant learning
//     rate that keeps the optimizer moving around the basin instead of
//     settling.
//   - EMA, an exponential moving average that decays old snapshots
//     geometrically. It tracks a moving trajectory rather than a
//     stationary one, which is what you want when the learning rate is
//     still decaying and later weights are genuinely better than earlier
//     ones. Decay values near 1 (0.999 is typical) average over roughly
//     1/(1-decay) recent snapshots.
//
// Unlike Lookahead, which feeds its averaged weights back into training,
// this wrapper never perturbs the live weights; the average is a separate
// candidate model you Apply once training is done (usually to a Copy of
// the network, so the trained weights survive for comparison).
type WeightAveragerOptimizer struct {
	Inner Optimizer

	// Decay is the EMA coefficient in (0, 1), or exactly 0 to take the
	// equal-weight SWA mean instead.
	Decay float64

	// StartAfter skips the first n invocations before averaging begins,
	// which keeps the noisy early trajectory out of the mean. Both variants
	// benefit; SWA in particular is normally started only once the learning
	// rate has reached its cyclic phase.
	StartAfter int

	avgW  []matrix.Matrix
	avgB  []matrix.Matrix
	step  int
	count int
}

// NewWeightAverager wraps inner with a running weight average. decay must
// be in [0, 1): 0 selects the equal-weight SWA mean, anything larger is the
// EMA coefficient. startAfter, which must not be negative, is the number of
// leading invocations to ignore.
func NewWeightAverager(inner Optimizer, decay float64, startAfter int) *WeightAveragerOptimizer {
	if decay < 0 || decay >= 1 {
		panic("goneural: weight averaging decay must be in [0, 1)")
	}
	if startAfter < 0 {
		panic("goneural: weight averaging startAfter must not be negative")
	}

	return &WeightAveragerOptimizer{
		Inner:      inner,
		Decay:      decay,
		StartAfter: startAfter,
	}
}

// SWA returns a weight averager taking the equal-weight mean of every
// snapshot after the first startAfter invocations.
func SWA(inner Optimizer, startAfter int) *WeightAveragerOptimizer {
	return NewWeightAverager(inner, 0, startAfter)
}

// EMA returns a weight averager taking an exponential moving average with
// the given decay, which must be in (0, 1).
func EMA(inner Optimizer, decay float64) *WeightAveragerOptimizer {
	if decay <= 0 {
		panic("goneural: EMA decay must be in (0, 1)")
	}
	return NewWeightAverager(inner, decay, 0)
}

// Optimize implements the Optimizer signature: it runs the inner optimizer
// and folds the resulting weights into the running average.
func (o *WeightAveragerOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	err := o.Inner(n, dataSet)

	o.step++
	if o.step <= o.StartAfter {
		return err
	}

	// The first snapshot after the warm-up is the average, for both
	// variants -- an EMA seeded from zeros would spend its first several
	// hundred steps crawling out of the origin, and the SWA mean of one
	// point is that point.
	if o.avgW == nil {
		o.avgW = make([]matrix.Matrix, len(n.Weights))
		o.avgB = make([]matrix.Matrix, len(n.Biases))
		for l := range n.Weights {
			o.avgW[l] = n.Weights[l].Copy()
			o.avgB[l] = n.Biases[l].Copy()
		}
		o.count = 1
		return err
	}

	o.count++

	// Both updates are "move the average a fraction of the way toward the
	// new point"; only the fraction differs. EMA uses a fixed 1 - decay,
	// while the equal-weight mean uses 1/count, which is exactly the
	// incremental form of summing everything and dividing at the end.
	rate := 1 - o.Decay
	if o.Decay == 0 {
		rate = 1 / float64(o.count)
	}

	for l := range n.Weights {
		o.avgW[l] = o.avgW[l].AddMatrix(n.Weights[l].SubtractMatrix(o.avgW[l]).Scale(rate))
		o.avgB[l] = o.avgB[l].AddMatrix(n.Biases[l].SubtractMatrix(o.avgB[l]).Scale(rate))
	}

	return err
}

// Count reports how many snapshots have gone into the average so far.
func (o *WeightAveragerOptimizer) Count() int {
	return o.count
}

// Apply copies the averaged weights and biases into the network, replacing
// whatever the inner optimizer last left there, and reports whether an
// average existed to copy (it does not once StartAfter has swallowed every
// invocation so far). Call it after training; to keep the trained weights
// around for comparison, apply to a Copy of the network instead.
func (o *WeightAveragerOptimizer) Apply(n *NeuralNetwork) bool {
	if o.avgW == nil {
		return false
	}
	if len(o.avgW) != len(n.Weights) {
		panic("goneural: weight averager applied to a network of a different shape")
	}

	for l := range o.avgW {
		n.Weights[l] = o.avgW[l].Copy()
		n.Biases[l] = o.avgB[l].Copy()
	}

	return true
}
