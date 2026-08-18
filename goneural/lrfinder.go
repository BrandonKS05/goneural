package goneural

import "math"

// LRProbe is one point of an LRFinder sweep: the learning rate that was
// tried and the mean per-sample loss the network reached under it.
type LRProbe struct {
	LearningRate float64
	Loss         float64
}

// LRSweep is the curve an LRFinder traces out, ordered from the smallest
// learning rate to the largest.
type LRSweep []LRProbe

// LRFinder implements the learning-rate range test (Smith, 2015): train a
// throwaway copy of the network while ramping the learning rate
// geometrically from Min to Max, and watch what the loss does. The result
// is the one plot that answers "what learning rate should I use?" without
// a grid search -- and the answer matters more than most hyperparameters,
// since too small wastes epochs crawling and too large diverges outright.
//
// The curve has a characteristic shape: a flat stretch where the rate is
// too small to move anything, a downward slope where training actually
// works, a minimum, and then a sharp climb as the steps overshoot. The
// useful rate is not the minimum -- by then training is already close to
// unstable -- but the steepest part of the descent, which is what Suggest
// returns.
//
// The sweep steps the rate once per optimizer invocation (one pass over
// the data) rather than once per mini-batch as the paper does, which is
// what this package's Optimizer signature exposes; keep the data set small
// or the step count modest to keep the test cheap. It costs one full
// training run, and it is worth it.
type LRFinder struct {
	// Optimizer is the optimizer under test, and SetLearningRate is the
	// hook that points it at a new rate -- the same pairing WithScheduleFunc
	// uses, since an Optimizer is an opaque function that may or may not
	// carry its own rate.
	Optimizer       Optimizer
	SetLearningRate func(lr float64)

	// Min and Max bound the geometric ramp, and Steps is how many rates to
	// try between them, inclusive of both ends.
	Min   float64
	Max   float64
	Steps int

	// Smoothing is the EMA coefficient in [0, 1) applied to the loss curve
	// before it is interpreted. Epoch-to-epoch loss is noisy enough that a
	// raw curve's steepest point is often just its unluckiest sample; 0.8
	// is a reasonable default and 0 disables smoothing.
	Smoothing float64

	// EpochsPerStep is how many passes over the data each probe gets before
	// the loss is read and the rate stepped up. One pass is enough on a
	// large data set, where a single epoch is already thousands of updates;
	// small sets need a handful for a rate to show what it can do. Defaults
	// to 1.
	EpochsPerStep int

	// DivergeFactor stops the sweep early once the smoothed loss climbs to
	// this multiple of the best value seen, since everything past that
	// point is a diverged network and tells you nothing. Defaults to 4;
	// a non-positive value disables the check.
	DivergeFactor float64
}

// NewLRFinder builds a range test over [min, max] in the given number of
// steps, with the default smoothing and divergence cutoff. min must be
// positive and strictly below max (the ramp is geometric, so it cannot
// start at or below zero), and steps must be at least 2.
func NewLRFinder(optimizer Optimizer, setLearningRate func(lr float64), min, max float64, steps int) *LRFinder {
	if min <= 0 || max <= min {
		panic("goneural: LRFinder needs 0 < min < max")
	}
	if steps < 2 {
		panic("goneural: LRFinder needs at least two steps")
	}
	if setLearningRate == nil {
		panic("goneural: LRFinder needs a SetLearningRate hook")
	}

	return &LRFinder{
		Optimizer:       optimizer,
		SetLearningRate: setLearningRate,
		Min:             min,
		Max:             max,
		Steps:           steps,
		EpochsPerStep:   1,
		Smoothing:       0.8,
		DivergeFactor:   4,
	}
}

// Run performs the sweep and returns the curve. The network is left
// untouched: the ramp trains a Copy, since the whole point is to push the
// weights until they blow up. The optimizer, on the other hand, is *not*
// copied -- it cannot be, being a function -- so any state it carries
// (momentum, Adam's moments) is left warmed up by the sweep. Build a fresh
// optimizer for the real training run.
//
// Note that the loss is compared across different learning rates, so a
// shuffled data set makes consecutive probes noisier; Run deliberately
// does not shuffle.
func (f *LRFinder) Run(n *NeuralNetwork, dataSet DataSet) LRSweep {
	if f.Steps < 2 || f.Min <= 0 || f.Max <= f.Min {
		panic("goneural: LRFinder needs at least two steps and 0 < Min < Max")
	}
	if len(dataSet) == 0 {
		panic("goneural: LRFinder needs a non-empty data set")
	}
	if f.Smoothing < 0 || f.Smoothing >= 1 {
		panic("goneural: LRFinder smoothing must be in [0, 1)")
	}

	probe := n.Copy()
	sweep := make(LRSweep, 0, f.Steps)

	// A geometric ramp, because what matters about a learning rate is its
	// order of magnitude: equal steps in log space give every decade the
	// same number of probes.
	ratio := math.Pow(f.Max/f.Min, 1/float64(f.Steps-1))

	lr := f.Min
	best, smoothed := math.Inf(1), 0.0

	for i := 0; i < f.Steps; i++ {
		f.SetLearningRate(lr)
		for e := 0; e < f.epochsPerStep(); e++ {
			f.Optimizer(probe, dataSet)
		}

		// Measured after the step, not during it: what the range test wants
		// to know is where the weights ended up, and an optimizer's own
		// return value is the loss it saw on the way there -- under the
		// previous probe's weights as much as this one's.
		loss := probe.MeanLoss(dataSet)
		sweep = append(sweep, LRProbe{LearningRate: lr, Loss: loss})

		// A diverged network can produce NaN or Inf outright, which is as
		// clear a stop signal as the ratio test.
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			break
		}

		if i == 0 {
			smoothed = loss
		} else {
			smoothed = f.Smoothing*smoothed + (1-f.Smoothing)*loss
		}
		if smoothed < best {
			best = smoothed
		}
		if f.DivergeFactor > 0 && smoothed > f.DivergeFactor*best {
			break
		}

		lr *= ratio
	}

	return sweep
}

// epochsPerStep treats an unset (zero) count as the default of one, so a
// zero-valued LRFinder built by hand still runs.
func (f *LRFinder) epochsPerStep() int {
	if f.EpochsPerStep < 1 {
		return 1
	}
	return f.EpochsPerStep
}

// Best returns the probe with the lowest loss, and whether the sweep had
// any usable (finite-loss) probe at all. This is the bottom of the valley,
// which is already too aggressive to train at -- see Suggest.
func (s LRSweep) Best() (LRProbe, bool) {
	best, found := LRProbe{}, false
	for _, p := range s {
		if math.IsNaN(p.Loss) || math.IsInf(p.Loss, 0) {
			continue
		}
		if !found || p.Loss < best.Loss {
			best, found = p, true
		}
	}
	return best, found
}

// Suggest returns the learning rate to actually train with: the point of
// steepest descent on the smoothed curve, measured per decade of learning
// rate, which is where the loss is falling fastest and the optimizer still
// has margin before the wall. It reports false when the sweep is too short
// or too degenerate to have a descending stretch -- typically because every
// rate tried was already past the cliff, in which case sweep a lower range.
//
// smoothing is the EMA coefficient applied before differentiating, in
// [0, 1); pass the finder's own Smoothing for consistency with the run.
func (s LRSweep) Suggest(smoothing float64) (float64, bool) {
	if smoothing < 0 || smoothing >= 1 {
		panic("goneural: LRSweep smoothing must be in [0, 1)")
	}

	usable := make(LRSweep, 0, len(s))
	for _, p := range s {
		if !math.IsNaN(p.Loss) && !math.IsInf(p.Loss, 0) {
			usable = append(usable, p)
		}
	}
	if len(usable) < 2 {
		return 0, false
	}

	smoothed := make([]float64, len(usable))
	smoothed[0] = usable[0].Loss
	for i := 1; i < len(usable); i++ {
		smoothed[i] = smoothing*smoothed[i-1] + (1-smoothing)*usable[i].Loss
	}

	// The gradient is taken against log(lr), not lr: on a geometric ramp
	// the raw spacing between probes grows with the rate, which would make
	// every late step look steep no matter what the loss did.
	steepest, at, found := 0.0, 0.0, false
	for i := 1; i < len(usable); i++ {
		dx := math.Log10(usable[i].LearningRate) - math.Log10(usable[i-1].LearningRate)
		if dx == 0 {
			continue
		}

		slope := (smoothed[i] - smoothed[i-1]) / dx
		if slope < steepest {
			steepest, at, found = slope, usable[i-1].LearningRate, true
		}
	}

	return at, found
}
