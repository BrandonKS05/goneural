package goneural

// SnapshotOptimizer wraps an inner optimizer and keeps a copy of the
// network every `every` invocations, collecting an Ensemble over the
// course of one training run.
//
// Bagging an ensemble costs one full training run per member. Snapshot
// ensembling (Huang et al., 2017) gets them for free, by exploiting what a
// cyclic learning-rate schedule already does: each time the rate is
// annealed down, training settles into some minimum; each time it jumps
// back up on a warm restart, training is kicked out of that minimum and
// goes looking for another. Photographing the weights at the bottom of
// every cycle yields models that reached genuinely different solutions --
// which is the only property that makes averaging them worth anything --
// for the price of a single run.
//
// The pairing that matters is with CosineAnnealing: set the snapshot
// interval equal to the schedule's period, so each copy is taken at a
// cycle's minimum rather than somewhere up the slope.
//
//	o := goneural.NewNadamOptimizer(16, 0.01)
//	optimizer := goneural.WithScheduleFunc(o.Optimize,
//		goneural.CosineAnnealing(0.01, 0.0001, 50),
//		func(lr float64) { o.LearningRate = lr })
//
//	snapshots := goneural.NewSnapshotOptimizer(optimizer, 50)
//	snapshots.WarmUp = 100 // let the first two cycles go by untaken
//
//	g.Train(snapshots.Optimize, data, 500)
//	fmt.Println(snapshots.Ensemble().Accuracy(test))
type SnapshotOptimizer struct {
	Inner Optimizer

	// Every is how many invocations pass between snapshots.
	Every int

	// WarmUp suppresses snapshots for this many leading invocations. The
	// first cycle or two usually end somewhere mediocre, and a bad member
	// drags an average down as surely as a good one lifts it.
	WarmUp int

	// Limit caps how many snapshots are kept; once full, the oldest is
	// dropped. Zero keeps every snapshot. Late members are generally the
	// better ones, so discarding from the front is the right end.
	Limit int

	members Ensemble
	step    int
}

// NewSnapshotOptimizer wraps inner, taking a snapshot every `every`
// invocations, which must be positive.
func NewSnapshotOptimizer(inner Optimizer, every int) *SnapshotOptimizer {
	if every < 1 {
		panic("goneural: snapshots need a positive interval")
	}

	return &SnapshotOptimizer{Inner: inner, Every: every}
}

// Optimize implements the Optimizer signature: it runs the inner optimizer
// and, on the right invocations, photographs the network.
func (o *SnapshotOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	if o.Every < 1 {
		panic("goneural: snapshots need a positive interval")
	}
	if o.WarmUp < 0 || o.Limit < 0 {
		panic("goneural: snapshot warm-up and limit must not be negative")
	}

	err := o.Inner(n, dataSet)

	o.step++
	if o.step <= o.WarmUp {
		return err
	}

	// Counted from the end of the warm-up, so the interval lines up with
	// the schedule's cycles rather than with an arbitrary offset.
	if (o.step-o.WarmUp)%o.Every != 0 {
		return err
	}

	o.members = append(o.members, n.Copy())
	if o.Limit > 0 && len(o.members) > o.Limit {
		o.members = o.members[len(o.members)-o.Limit:]
	}

	return err
}

// Ensemble returns the snapshots collected so far, in the order they were
// taken. The result shares the stored networks rather than copying them,
// so training on the live network afterwards leaves them untouched -- they
// were copies at the moment they were taken.
func (o *SnapshotOptimizer) Ensemble() Ensemble {
	return append(Ensemble(nil), o.members...)
}

// Count reports how many snapshots have been taken.
func (o *SnapshotOptimizer) Count() int {
	return len(o.members)
}
