package goneural

import (
	"math"
	"math/rand"
)

// Scaler rescales input features by an affine per-feature map,
// (x - Center) / Spread, with the constants fitted once on the training
// set. Feeding raw features of wildly different magnitudes to a network is
// a common and avoidable way to cripple training: a feature measured in
// thousands dominates the weighted sum, saturates sigmoid and tanh units,
// and forces the learning rate down to whatever the largest feature can
// tolerate. Rescaling puts every input on comparable footing.
//
// Fit on the training split only, then Transform both splits with that same
// Scaler. Fitting on the full data set before splitting leaks test
// statistics into training and quietly inflates the reported score.
type Scaler struct {
	Center []float64
	Spread []float64
}

// FitStandardizer fits a zero-mean, unit-variance ("z-score") scaler:
// Center is each feature's mean and Spread its population standard
// deviation. This is the usual default -- it makes no assumption about
// bounds and handles unbounded features gracefully, though it does not
// confine outputs to a fixed range.
func FitStandardizer(dataSet DataSet) *Scaler {
	features := fitFeatureCount(dataSet)

	mean := make([]float64, features)
	for _, ds := range dataSet {
		for i, v := range ds.Inputs {
			mean[i] += v
		}
	}
	for i := range mean {
		mean[i] /= float64(len(dataSet))
	}

	spread := make([]float64, features)
	for _, ds := range dataSet {
		for i, v := range ds.Inputs {
			spread[i] += (v - mean[i]) * (v - mean[i])
		}
	}
	for i := range spread {
		spread[i] = safeSpread(math.Sqrt(spread[i] / float64(len(dataSet))))
	}

	return &Scaler{Center: mean, Spread: spread}
}

// FitMinMaxScaler fits a scaler that maps each feature's observed range
// onto [0, 1]: Center is the minimum and Spread the min-to-max distance.
// Useful when features have genuine hard bounds (pixel intensities, survey
// scales), and when you want inputs to land in the range sigmoid units are
// happiest with. Its weakness is the mirror image of the standardizer's
// strength: one extreme outlier sets the range and squashes everything
// else into a sliver of it, and values beyond the fitted range escape
// [0, 1] rather than being clipped.
func FitMinMaxScaler(dataSet DataSet) *Scaler {
	features := fitFeatureCount(dataSet)

	min := make([]float64, features)
	max := make([]float64, features)
	copy(min, dataSet[0].Inputs)
	copy(max, dataSet[0].Inputs)

	for _, ds := range dataSet[1:] {
		for i, v := range ds.Inputs {
			if v < min[i] {
				min[i] = v
			}
			if v > max[i] {
				max[i] = v
			}
		}
	}

	spread := make([]float64, features)
	for i := range spread {
		spread[i] = safeSpread(max[i] - min[i])
	}

	return &Scaler{Center: min, Spread: spread}
}

// fitFeatureCount returns the shared input width of the data set, panicking
// on an empty set (nothing to fit) or ragged samples (no shared width).
func fitFeatureCount(dataSet DataSet) int {
	if len(dataSet) == 0 {
		panic("goneural: cannot fit a scaler on an empty data set")
	}

	features := len(dataSet[0].Inputs)
	for _, ds := range dataSet {
		if len(ds.Inputs) != features {
			panic("goneural: cannot fit a scaler on samples of differing widths")
		}
	}

	return features
}

// safeSpread keeps a constant feature -- zero variance, or a zero-width
// range -- from dividing by zero. Such a feature carries no information at
// all, so mapping it to a constant 0 (which a spread of 1 does, since the
// value equals its own center) loses nothing.
func safeSpread(spread float64) float64 {
	if spread == 0 {
		return 1
	}
	return spread
}

// Transform returns a copy of the data set with every sample's inputs
// rescaled. Targets are carried through untouched, and the receiver's
// samples are left alone, so the raw data stays available.
func (s *Scaler) Transform(dataSet DataSet) DataSet {
	out := make(DataSet, len(dataSet))
	for i, ds := range dataSet {
		out[i] = DataSample{
			Inputs:  s.TransformInputs(ds.Inputs),
			Targets: ds.Targets,
		}
	}
	return out
}

// TransformInputs rescales a single input vector, which is what you want at
// prediction time: live inputs must go through the same map the training
// data did before reaching Predict.
func (s *Scaler) TransformInputs(inputs []float64) []float64 {
	if len(inputs) != len(s.Center) {
		panic("goneural: scaler applied to inputs of the wrong width")
	}

	out := make([]float64, len(inputs))
	for i, v := range inputs {
		out[i] = (v - s.Center[i]) / s.Spread[i]
	}
	return out
}

// InverseInputs undoes TransformInputs, mapping a rescaled vector back to
// the original units. Constant features, whose spread was replaced by 1,
// come back as their center.
func (s *Scaler) InverseInputs(inputs []float64) []float64 {
	if len(inputs) != len(s.Center) {
		panic("goneural: scaler applied to inputs of the wrong width")
	}

	out := make([]float64, len(inputs))
	for i, v := range inputs {
		out[i] = v*s.Spread[i] + s.Center[i]
	}
	return out
}

// Mixup returns a fresh data set of the same size in which every sample is
// a convex combination of two real ones (Zhang et al., 2018): inputs and
// targets are blended with the *same* random weight lambda, drawn from a
// Beta(alpha, alpha) distribution. Training on the blends -- rather than on
// the originals -- asks the network to behave linearly between its training
// points instead of memorizing them, which is a surprisingly effective
// regularizer and noticeably tames overconfident predictions.
//
// alpha controls how aggressive the blending is, and must be positive.
// Small values (0.2 is the paper's image-classification default) keep
// lambda near 0 or 1, so most samples are nearly untouched originals;
// alpha = 1 makes lambda uniform on [0, 1]; large values pull lambda toward
// an even 50/50 mix of two samples, which is usually too much.
//
// It requires targets you can meaningfully average -- one-hot class labels
// or regression values, not hard class indices -- and pairs naturally with
// re-drawing every epoch rather than mixing once up front.
func (t DataSet) Mixup(alpha float64) DataSet {
	if alpha <= 0 {
		panic("goneural: Mixup alpha must be positive")
	}
	if len(t) == 0 {
		return DataSet{}
	}
	// Checked up front rather than per blend, since which samples get
	// paired is random: a ragged set must fail the same way every run.
	for _, ds := range t {
		if len(ds.Inputs) != len(t[0].Inputs) || len(ds.Targets) != len(t[0].Targets) {
			panic("goneural: Mixup needs samples of a uniform width")
		}
	}

	out := make(DataSet, len(t))
	for i, ds := range t {
		other := t[rand.Intn(len(t))]
		lambda := betaSample(alpha, alpha)

		out[i] = DataSample{
			Inputs:  blend(ds.Inputs, other.Inputs, lambda),
			Targets: blend(ds.Targets, other.Targets, lambda),
		}
	}

	return out
}

// blend returns lambda*a + (1-lambda)*b elementwise.
func blend(a, b []float64, lambda float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = lambda*a[i] + (1-lambda)*b[i]
	}
	return out
}

// betaSample draws from Beta(a, b) using the standard ratio of two Gamma
// draws: if X ~ Gamma(a, 1) and Y ~ Gamma(b, 1) then X/(X+Y) ~ Beta(a, b).
func betaSample(a, b float64) float64 {
	x := gammaSample(a)
	y := gammaSample(b)
	if x+y == 0 { // both underflowed to zero; the ratio is undefined
		return 0.5
	}
	return x / (x + y)
}

// gammaSample draws from Gamma(shape, 1) by Marsaglia and Tsang's (2000)
// squeeze method, which is exact rather than approximate: it proposes
// d*(1 + c*normal)^3 and accepts against the Gamma density with a cheap
// pre-test that almost always short-circuits the logarithms. The method
// needs shape >= 1, so smaller shapes -- the interesting ones for mixup --
// go through the boost identity Gamma(s) = Gamma(s+1) * U^(1/s).
func gammaSample(shape float64) float64 {
	if shape < 1 {
		return gammaSample(shape+1) * math.Pow(rand.Float64(), 1/shape)
	}

	d := shape - 1.0/3.0
	c := 1 / math.Sqrt(9*d)

	for {
		var v, x float64
		for v <= 0 {
			x = rand.NormFloat64()
			v = 1 + c*x
		}
		v = v * v * v

		u := rand.Float64()
		if u < 1-0.0331*x*x*x*x { // the squeeze: accept without logs
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v
		}
	}
}
