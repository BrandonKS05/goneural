package goneural

import "math/rand"

// Ensemble is a committee of networks polled together. Averaging several
// independently trained models is the least glamorous way to gain accuracy
// and one of the most reliable: each network's errors come partly from
// where its random initialization happened to land, and those idiosyncratic
// errors do not line up across members, while the signal does. What
// survives the average is mostly what the members agree on.
//
// The members need not share an architecture -- only an input and output
// width, so their predictions can be pooled.
type Ensemble []*NeuralNetwork

// Predict polls every member and returns the elementwise mean of their
// outputs, which is "soft voting": a member that is unsure contributes a
// hedged vote rather than a full one. On a softmax classifier the average
// of several distributions is itself a distribution, so the result stays
// interpretable as class probabilities. It panics on an empty ensemble or
// on members whose outputs are of differing widths.
func (e Ensemble) Predict(inputs []float64) []float64 {
	if len(e) == 0 {
		panic("goneural: cannot predict with an empty ensemble")
	}

	var out []float64
	for _, member := range e {
		prediction := member.Predict(inputs)
		if out == nil {
			out = make([]float64, len(prediction))
		}
		if len(prediction) != len(out) {
			panic("goneural: ensemble members disagree on the output width")
		}

		for i, v := range prediction {
			out[i] += v
		}
	}

	for i := range out {
		out[i] /= float64(len(e))
	}
	return out
}

// Vote returns the class index the ensemble picks by majority ("hard
// voting"): each member casts one whole vote for its own top class, ties
// going to the lowest index. It is the blunter counterpart to averaging
// through Predict -- worth reaching for when the members' output scales are
// not comparable (differently calibrated confidences, or a mix of losses),
// since a hard vote only cares about each member's ranking.
func (e Ensemble) Vote(inputs []float64) int {
	if len(e) == 0 {
		panic("goneural: cannot vote with an empty ensemble")
	}

	votes := map[int]int{}
	for _, member := range e {
		votes[classOf(member.Predict(inputs))]++
	}

	winner, best := 0, -1
	for class, count := range votes {
		if count > best || (count == best && class < winner) {
			winner, best = class, count
		}
	}
	return winner
}

// Accuracy is the fraction of the data set the ensemble's averaged
// prediction classifies correctly, derived exactly the way a single
// network's Accuracy is. An empty data set has an accuracy of 0.
func (e Ensemble) Accuracy(dataSet DataSet) float64 {
	if len(dataSet) == 0 {
		return 0
	}

	correct := 0
	for _, ds := range dataSet {
		if classOf(e.Predict(ds.Inputs)) == classOf(ds.Targets) {
			correct++
		}
	}

	return float64(correct) / float64(len(dataSet))
}

// Bootstrap returns a resample of the data set: the same number of samples,
// drawn uniformly *with replacement*, so roughly a third of the originals
// are missing from any given draw and others appear two or three times.
// This is the resampling that makes bagged members differ from one another
// in what they have seen, not just in where they started. The samples
// themselves are shared, not copied.
func (t DataSet) Bootstrap() DataSet {
	out := make(DataSet, len(t))
	for i := range out {
		out[i] = t[rand.Intn(len(t))]
	}
	return out
}

// Bag trains an ensemble of the given size by bootstrap aggregating: every
// member is a freshly initialized network with the prototype's shape,
// trained by the supplied function on its own bootstrap resample of the
// data. Members differ in both their starting weights and their training
// data, which is what decorrelates their mistakes enough for the average to
// beat any one of them.
//
// The prototype is used only as a template -- its own weights are never
// touched and never trained. train receives a member and that member's
// resampled data, and is where the optimizer, epoch count and any
// per-member configuration go:
//
//	ensemble := goneural.Bag(prototype, data, 5, func(n *goneural.NeuralNetwork, d goneural.DataSet) {
//		n.Train(goneural.Adam(8), d, 200)
//	})
//
// It panics unless members is positive and the data set is non-empty.
func Bag(prototype *NeuralNetwork, dataSet DataSet, members int, train func(n *NeuralNetwork, dataSet DataSet)) Ensemble {
	if members < 1 {
		panic("goneural: Bag needs at least one member")
	}
	if len(dataSet) == 0 {
		panic("goneural: Bag needs a non-empty data set")
	}
	if train == nil {
		panic("goneural: Bag needs a training function")
	}

	ensemble := make(Ensemble, members)
	for i := range ensemble {
		// New, not Copy: a copy would start every member from the same
		// weights, and identical starting points on overlapping data give
		// correlated members and a pointless average.
		member := New(prototype.LearningRate, prototype.Loss, prototype.Layers...)
		member.HiddenDropout = prototype.HiddenDropout

		train(member, dataSet.Bootstrap())
		ensemble[i] = member
	}

	return ensemble
}
