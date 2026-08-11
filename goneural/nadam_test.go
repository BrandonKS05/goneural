package goneural

import (
	"math"
	"math/rand"
	"testing"
)

func TestNadamReducesLoss(t *testing.T) {
	rand.Seed(7)
	testOptimizerLearnsXOR(t, Nadam(2, 0.05), 500)
}

// TestNadamStepAnticipatesMomentum pins the Nesterov part: on the very
// first update from rest, Nadam's numerator folds in an extra helping of
// the current gradient, so its step must be strictly larger than Adam's for
// the same input.
func TestNadamStepAnticipatesMomentum(t *testing.T) {
	const (
		lr      = 0.1
		beta1   = 0.9
		beta2   = 0.999
		epsilon = 1e-8
	)
	bc1 := 1 - beta1
	bc2 := 1 - beta2

	grad := []float64{0.5}

	adam := adamStep([]float64{grad[0]}, make([]float64, 1), make([]float64, 1), 1, lr, beta1, beta2, epsilon, bc1, bc2)[0]
	nadam := nadamStep([]float64{grad[0]}, make([]float64, 1), make([]float64, 1), 1, lr, beta1, beta2, epsilon, bc1, bc2)[0]

	if nadam <= adam {
		t.Fatalf("expected Nadam's first step (%g) to exceed Adam's (%g)", nadam, adam)
	}

	// Both must still step in the gradient's direction.
	if math.Signbit(nadam) != math.Signbit(grad[0]) {
		t.Fatalf("Nadam stepped against the gradient: %g", nadam)
	}
}
