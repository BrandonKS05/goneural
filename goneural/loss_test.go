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
	for _, loss := range []Loss{MSE(), CrossEntropy(), MAE(), Huber(), LogCosh()} {
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
