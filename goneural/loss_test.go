package goneural

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

// TestLossFPrimeMatchesNumericGradient checks MAE and Huber against a
// central-difference gradient of their own F. The package convention is
// that FPrime returns the *negative* gradient with respect to the
// prediction (the descent direction outputDelta adds), so each element must
// equal -dF/dpred. Probe points stay away from the losses' kinks (0 for
// MAE, +-delta for Huber), where the numeric estimate is meaningless.
func TestLossFPrimeMatchesNumericGradient(t *testing.T) {
	const h = 1e-7

	preds := []float64{0.9, -1.6, 0.3}
	targets := []float64{0.2, 0.4, 0.8}

	for _, tc := range []struct {
		name string
		loss Loss
	}{
		{"mae", MAE()},
		{"huber", Huber()},
		{"huber_delta", HuberWithDelta(0.4)},
		{"log_cosh", LogCosh()},
	} {
		target := matrix.NewFromArray(targets)

		for k := range preds {
			bumped := func(offset float64) float64 {
				p := append([]float64(nil), preds...)
				p[k] += offset
				return tc.loss.F(matrix.NewFromArray(p), target)
			}

			numeric := (bumped(h) - bumped(-h)) / (2 * h)
			got := tc.loss.FPrime(matrix.NewFromArray(preds), target).Flatten()[k]

			if math.Abs(got-(-numeric)) > 1e-5 {
				t.Errorf("%s: FPrime[%d] = %g, want -dF/dpred = %g", tc.name, k, got, -numeric)
			}
		}
	}
}

// TestLossNameRoundTrip guards Save/Load: every named loss must come back
// out of the registry under the same name.
func TestLossNameRoundTrip(t *testing.T) {
	for _, loss := range []Loss{MSE(), CrossEntropy(), MAE(), Huber(), LogCosh(), BinaryCrossEntropy(), Hinge()} {
		got := getLossFromname(loss.Name)
		if got.Name != loss.Name {
			t.Errorf("getLossFromname(%q).Name = %q", loss.Name, got.Name)
		}
	}
}

func TestHuberLearnsXOR(t *testing.T) {
	rand.Seed(7)

	n := New(0.9, Huber(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	data := xorData()
	optimizer := MBGD(2)
	firstErr := optimizer(n, data)

	var lastErr float64
	for i := 0; i < 3000; i++ {
		lastErr = optimizer(n, data)
	}

	if lastErr >= firstErr {
		t.Fatalf("expected training to reduce loss, got firstErr=%f lastErr=%f", firstErr, lastErr)
	}

	for _, ds := range data {
		got := n.Predict(ds.Inputs)[0]
		want := ds.Targets[0]
		if diff := got - want; diff > 0.2 || diff < -0.2 {
			t.Errorf("Predict(%v) = %f, want close to %f", ds.Inputs, got, want)
		}
	}
}

// TestClassificationLossFPrimeMatchesNumericGradient checks the two
// classification losses against a central difference of their own F. They
// need their own probe points: binary cross-entropy is only defined on
// (0, 1), and hinge is kinked exactly at a margin of 1.
func TestClassificationLossFPrimeMatchesNumericGradient(t *testing.T) {
	const h = 1e-6

	for _, tc := range []struct {
		name    string
		loss    Loss
		preds   []float64
		targets []float64
	}{
		{"binary_cross_entropy", BinaryCrossEntropy(), []float64{0.7, 0.2, 0.55}, []float64{1, 0, 1}},
		{"hinge_violated", Hinge(), []float64{0.3, -0.2, 0.5}, []float64{1, 1, -1}},
		{"hinge_satisfied", Hinge(), []float64{2.5, 1.4, -3.0}, []float64{1, 1, -1}},
	} {
		target := matrix.NewFromArray(tc.targets)

		for k := range tc.preds {
			bumped := func(offset float64) float64 {
				p := append([]float64(nil), tc.preds...)
				p[k] += offset
				return tc.loss.F(matrix.NewFromArray(p), target)
			}

			numeric := (bumped(h) - bumped(-h)) / (2 * h)
			got := tc.loss.FPrime(matrix.NewFromArray(tc.preds), target).Flatten()[k]

			if math.Abs(got-(-numeric)) > 1e-5 {
				t.Errorf("%s: FPrime[%d] = %g, want -dF/dpred = %g", tc.name, k, got, -numeric)
			}
		}
	}
}

// TestBinaryCrossEntropyStaysFinite pins the clamping: a fully saturated
// prediction on the wrong side must produce a large but finite loss and
// gradient rather than infinities that poison every later weight.
func TestBinaryCrossEntropyStaysFinite(t *testing.T) {
	loss := BinaryCrossEntropy()

	pred := matrix.NewFromArray([]float64{0, 1})
	target := matrix.NewFromArray([]float64{1, 0})

	if got := loss.F(pred, target); math.IsInf(got, 0) || math.IsNaN(got) {
		t.Errorf("F on saturated predictions = %g, want finite", got)
	}
	for i, got := range loss.FPrime(pred, target).Flatten() {
		if math.IsInf(got, 0) || math.IsNaN(got) {
			t.Errorf("FPrime[%d] on saturated predictions = %g, want finite", i, got)
		}
	}
}

// TestBinaryCrossEntropyRewardsConfidence sanity-checks the loss surface:
// a confidently correct prediction must cost less than a hedged one, which
// in turn must cost less than a confidently wrong one.
func TestBinaryCrossEntropyRewardsConfidence(t *testing.T) {
	loss := BinaryCrossEntropy()
	target := matrix.NewFromArray([]float64{1})

	confident := loss.F(matrix.NewFromArray([]float64{0.99}), target)
	hedged := loss.F(matrix.NewFromArray([]float64{0.5}), target)
	wrong := loss.F(matrix.NewFromArray([]float64{0.01}), target)

	if !(confident < hedged && hedged < wrong) {
		t.Errorf("losses not ordered: confident %g, hedged %g, wrong %g", confident, hedged, wrong)
	}
}

// TestHingeIgnoresSamplesPastTheMargin is the property that separates hinge
// from the log losses: once a prediction clears the margin it contributes
// nothing at all, no matter how far past it goes.
func TestHingeIgnoresSamplesPastTheMargin(t *testing.T) {
	loss := Hinge()
	target := matrix.NewFromArray([]float64{1})

	for _, pred := range []float64{1.0, 1.5, 40} {
		if got := loss.F(matrix.NewFromArray([]float64{pred}), target); got != 0 {
			t.Errorf("F(%g) = %g, want 0 past the margin", pred, got)
		}
		if got := loss.FPrime(matrix.NewFromArray([]float64{pred}), target).At(0, 0); got != 0 {
			t.Errorf("FPrime(%g) = %g, want 0 past the margin", pred, got)
		}
	}

	// Just inside the margin the sample is back in play.
	if got := loss.F(matrix.NewFromArray([]float64{0.9}), target); math.Abs(got-0.1) > 1e-12 {
		t.Errorf("F(0.9) = %g, want 0.1", got)
	}
}

// TestBinaryCrossEntropyLearnsXOR trains end to end through the sigmoid
// output layer the loss is meant for.
func TestBinaryCrossEntropyLearnsXOR(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, BinaryCrossEntropy(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	data := xorData()
	optimizer := Adam(2)
	for i := 0; i < 600; i++ {
		data.Shuffle()
		optimizer(n, data)
	}

	for _, ds := range data {
		if diff := n.Predict(ds.Inputs)[0] - ds.Targets[0]; math.Abs(diff) > 0.2 {
			t.Errorf("Predict(%v) = %f, want close to %f", ds.Inputs, n.Predict(ds.Inputs)[0], ds.Targets[0])
		}
	}
}

// TestHingeLearnsSeparableProblem trains a margin classifier on an
// AND-shaped problem with the -1/+1 targets hinge expects and an
// unsquashed output layer.
func TestHingeLearnsSeparableProblem(t *testing.T) {
	rand.Seed(7)

	n := New(0.05, Hinge(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Tanh()},
		Layer{Nodes: 1, Activator: Identity()},
	)

	data := DataSet{
		{Inputs: []float64{1, 1}, Targets: []float64{1}},
		{Inputs: []float64{1, 0}, Targets: []float64{-1}},
		{Inputs: []float64{0, 1}, Targets: []float64{-1}},
		{Inputs: []float64{0, 0}, Targets: []float64{-1}},
	}

	optimizer := Adam(2)
	for i := 0; i < 600; i++ {
		data.Shuffle()
		optimizer(n, data)
	}

	for _, ds := range data {
		got := n.Predict(ds.Inputs)[0]
		if got*ds.Targets[0] <= 0 {
			t.Errorf("Predict(%v) = %f, want the sign of %f", ds.Inputs, got, ds.Targets[0])
		}
	}
}
