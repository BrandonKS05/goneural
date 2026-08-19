package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// PCA is a fitted principal component analysis: an orthogonal change of
// basis that re-expresses the inputs along the directions they actually
// vary in, ordered from the most variance to the least. Keeping only the
// leading directions shrinks the input width while throwing away as little
// of the data's structure as possible.
//
// It earns its keep in three ways. It cuts the input layer down, which is
// the cheapest way to shrink a network. It decorrelates the features,
// which is exactly the condition plain gradient descent likes and rarely
// gets -- correlated inputs carve the loss surface into long narrow
// valleys that a single global learning rate has to crawl along. And with
// Whiten set it equalizes their scales too, so no direction dominates the
// weighted sum simply for being measured in larger units.
//
// The catch is that the directions are chosen by variance alone, with no
// reference to the targets: a low-variance direction that happens to be
// the one that separates the classes will be discarded just the same. Fit
// on the training split only, like any other Scaler, and check
// ExplainedVarianceRatio before trusting a reduction.
type PCA struct {
	// Mean is the per-feature center subtracted before projecting.
	Mean []float64

	// Components holds the principal axes as rows -- one row per retained
	// component, each a unit vector in the original feature space, ordered
	// by descending variance.
	Components matrix.Matrix

	// Variances is the variance the data has along each retained axis (the
	// covariance matrix's eigenvalues), and TotalVariance is the variance
	// over all axes, retained or not, which is what the retained ones are
	// a fraction of.
	Variances     []float64
	TotalVariance float64

	// Whiten additionally divides each projected coordinate by its own
	// standard deviation, leaving every component with unit variance. That
	// completes the job -- uncorrelated *and* commensurate inputs -- at the
	// cost of amplifying the trailing components, which are the ones most
	// likely to be noise.
	Whiten bool
}

// FitPCA fits a PCA to the data set's inputs, keeping the given number of
// components (which must be between 1 and the input width). It works from
// the eigendecomposition of the covariance matrix: its eigenvectors are
// the axes of the data's spread and the matching eigenvalues are how much
// spread each carries.
//
// The sign of an eigenvector is mathematically arbitrary -- an axis and
// its negation describe the same line -- so each component is normalized
// to have its largest-magnitude entry positive, which keeps a refit on the
// same data from silently flipping a feature's sign.
func FitPCA(dataSet DataSet, components int) *PCA {
	features := fitFeatureCount(dataSet)
	if components < 1 || components > features {
		panic("goneural: PCA component count out of range")
	}
	if len(dataSet) < 2 {
		panic("goneural: PCA needs at least two samples")
	}

	mean := make([]float64, features)
	for _, ds := range dataSet {
		for i, v := range ds.Inputs {
			mean[i] += v
		}
	}
	for i := range mean {
		mean[i] /= float64(len(dataSet))
	}

	// The covariance matrix, built directly rather than through a matrix
	// product: at these sizes the explicit double loop is clearer, and only
	// the upper triangle needs computing since the result is symmetric.
	covariance := matrix.New(features, features, nil)
	for _, ds := range dataSet {
		for i := 0; i < features; i++ {
			for j := i; j < features; j++ {
				c := (ds.Inputs[i] - mean[i]) * (ds.Inputs[j] - mean[j])
				covariance.Set(i, j, covariance.At(i, j)+c)
			}
		}
	}
	for i := 0; i < features; i++ {
		for j := i; j < features; j++ {
			v := covariance.At(i, j) / float64(len(dataSet))
			covariance.Set(i, j, v)
			covariance.Set(j, i, v)
		}
	}

	values, vectors, ok := covariance.SymmetricEigen()
	if !ok {
		panic("goneural: PCA could not decompose the covariance matrix")
	}

	total := 0.0
	for _, v := range values {
		// Rounding can push a zero-variance direction a hair below zero;
		// a covariance matrix has no genuinely negative eigenvalues.
		total += math.Max(v, 0)
	}

	axes := matrix.New(components, features, nil)
	variances := make([]float64, components)
	for k := 0; k < components; k++ {
		variances[k] = math.Max(values[k], 0)

		// Orient the axis deterministically before storing it.
		sign := 1.0
		largest := 0.0
		for f := 0; f < features; f++ {
			if v := vectors.At(f, k); math.Abs(v) > largest {
				largest, sign = math.Abs(v), math.Copysign(1, v)
			}
		}

		for f := 0; f < features; f++ {
			axes.Set(k, f, sign*vectors.At(f, k))
		}
	}

	return &PCA{
		Mean:          mean,
		Components:    axes,
		Variances:     variances,
		TotalVariance: total,
	}
}

// ExplainedVarianceRatio returns the fraction of the data's total variance
// each retained component accounts for. The entries sum to at most 1, and
// how close they come to it is the honest measure of what a reduction
// costs: three components explaining 0.98 is a bargain, three explaining
// 0.4 is throwing the data away.
func (p *PCA) ExplainedVarianceRatio() []float64 {
	out := make([]float64, len(p.Variances))
	if p.TotalVariance == 0 {
		return out
	}

	for i, v := range p.Variances {
		out[i] = v / p.TotalVariance
	}
	return out
}

// TransformInputs projects one input vector onto the retained components,
// returning a vector of that many coordinates. Use it at prediction time,
// exactly as with a Scaler: whatever map the training data went through,
// live inputs must go through too.
func (p *PCA) TransformInputs(inputs []float64) []float64 {
	if len(inputs) != len(p.Mean) {
		panic("goneural: PCA applied to inputs of the wrong width")
	}

	out := make([]float64, p.Components.Rows)
	for k := range out {
		sum := 0.0
		for f, v := range inputs {
			sum += p.Components.At(k, f) * (v - p.Mean[f])
		}

		if p.Whiten {
			// Guarded against a degenerate axis: a component with no
			// variance carries no information, and its coordinate is
			// already zero, so leaving it unscaled loses nothing.
			if sd := math.Sqrt(p.Variances[k]); sd > 1e-12 {
				sum /= sd
			}
		}
		out[k] = sum
	}

	return out
}

// InverseInputs maps a projected vector back into the original feature
// space. It is a genuine inverse only when every component was retained;
// otherwise it returns the closest point in the retained subspace, which
// is what makes PCA a lossy compression you can look at -- reconstruct a
// reduced sample and see what survived.
func (p *PCA) InverseInputs(inputs []float64) []float64 {
	if len(inputs) != p.Components.Rows {
		panic("goneural: PCA inverse applied to inputs of the wrong width")
	}

	out := append([]float64(nil), p.Mean...)
	for k, coordinate := range inputs {
		if p.Whiten {
			coordinate *= math.Sqrt(p.Variances[k])
		}
		for f := range out {
			out[f] += coordinate * p.Components.At(k, f)
		}
	}

	return out
}

// Transform returns a copy of the data set with every sample's inputs
// projected. Targets are carried through untouched and the receiver's
// samples are left alone, matching Scaler.Transform.
func (p *PCA) Transform(dataSet DataSet) DataSet {
	out := make(DataSet, len(dataSet))
	for i, ds := range dataSet {
		out[i] = DataSample{
			Inputs:  p.TransformInputs(ds.Inputs),
			Targets: ds.Targets,
		}
	}
	return out
}
