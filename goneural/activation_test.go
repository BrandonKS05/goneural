package goneural

import (
	"math"
	"math/rand"
	"testing"
)

// numericDerivative estimates f'(x) with a central difference.
func numericDerivative(f func(float64) float64, x float64) float64 {
	const h = 1e-6
	return (f(x+h) - f(x-h)) / (2 * h)
}

// TestFPrimeMatchesNumericDerivative pins the package's derivative
// convention: FPrime receives the activation's *output* y = F(x) and must
// return dF/dx at that x.
func TestFPrimeMatchesNumericDerivative(t *testing.T) {
	cases := []struct {
		name string
		act  Activation
	}{
		{"sigmoid", Sigmoid()},
		{"tanh", Tanh()},
		{"leaky_relu", LeakyReLU()},
		{"leaky_relu_alpha", LeakyReLUWithAlpha(0.2)},
		{"elu", ELU()},
		{"elu_alpha", ELUWithAlpha(0.5)},
		{"softplus", Softplus()},
	}

	for _, tc := range cases {
		for _, x := range []float64{-2.5, -1, -0.3, 0.4, 1.7} {
			got := tc.act.FPrime(tc.act.F(x))
			want := numericDerivative(tc.act.F, x)
			if math.Abs(got-want) > 1e-5 {
				t.Errorf("%s: FPrime(F(%g)) = %g, want %g", tc.name, x, got, want)
			}
		}
	}
}

// TestActivationNameRoundTrip guards Save/Load: every named activation must
// come back out of the registry under the same name.
func TestActivationNameRoundTrip(t *testing.T) {
	for _, act := range []Activation{Sigmoid(), ReLU(), Identity(), Tanh(), LeakyReLU(), ELU(), Softplus()} {
		got := getActivationFromName(act.Name)
		if got.Name != act.Name {
			t.Errorf("getActivationFromName(%q).Name = %q", act.Name, got.Name)
		}
	}
}

func TestTanhLearnsXOR(t *testing.T) {
	rand.Seed(3)

	n := New(0.5, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Tanh()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	data := xorData()
	optimizer := MBGD(2)
	firstErr := optimizer(n, data)

	var lastErr float64
	for i := 0; i < 1500; i++ {
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
