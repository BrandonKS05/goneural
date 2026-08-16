package goneural

import (
	"math"
	"math/rand"
	"testing"
)

func TestRAdamReducesLoss(t *testing.T) {
	rand.Seed(11)
	// RAdam spends its first steps on un-adapted momentum by design, so it
	// needs a longer budget than Adamax at the same learning rate.
	testOptimizerLearnsXOR(t, RAdam(2, 0.05), 1500)
}

// TestRAdamWithholdsAdaptivityEarly pins the behaviour that separates RAdam
// from Adam: for the first handful of steps the second moment is not just
// noisy but has undefined variance, and RAdam refuses to divide by it at all.
func TestRAdamWithholdsAdaptivityEarly(t *testing.T) {
	const beta2 = 0.999

	firstRectified := 0
	for step := 1; step <= 20; step++ {
		_, ok := radamRectification(radamRho(beta2, step))
		if ok {
			firstRectified = step
			break
		}
	}

	if firstRectified == 0 {
		t.Fatal("rectification never engaged within 20 steps")
	}

	// rho crosses 4 at step 5 for beta2=0.999, so steps 1-4 must be un-adapted.
	if firstRectified != 5 {
		t.Errorf("rectification engaged at step %d, want 5", firstRectified)
	}

	for step := 1; step < firstRectified; step++ {
		if _, ok := radamRectification(radamRho(beta2, step)); ok {
			t.Errorf("step %d rectified, want plain momentum", step)
		}
	}
}

// TestRAdamUnrectifiedStepIsPlainMomentum checks that the early phase really
// ignores the second moment: a huge v must not shrink the step the way it
// would under Adam.
func TestRAdamUnrectifiedStepIsPlainMomentum(t *testing.T) {
	const lr = 0.01

	m := []float64{0}
	v := []float64{1e6} // enormous second moment

	step := radamStep([]float64{2}, m, v, 1, lr, 0.9, 0.999, 1e-8, 1, 1, 0, false)

	// m becomes 0.9*0 + 0.1*2 = 0.2, so an un-adapted step is lr * 0.2.
	want := lr * 0.2
	if math.Abs(step[0]-want) > 1e-12 {
		t.Fatalf("unrectified step = %g, want %g (second moment must be ignored)", step[0], want)
	}
}

// TestRAdamRectificationApproachesOne confirms the correction fades out, so
// RAdam converges to ordinary Adam rather than permanently damping it.
func TestRAdamRectificationApproachesOne(t *testing.T) {
	const beta2 = 0.999

	early, ok := radamRectification(radamRho(beta2, 10))
	if !ok {
		t.Fatal("expected rectification to be active at step 10")
	}

	late, ok := radamRectification(radamRho(beta2, 100000))
	if !ok {
		t.Fatal("expected rectification to be active at step 100000")
	}

	if !(early < late) {
		t.Errorf("rectification did not increase: step 10 = %g, step 100000 = %g", early, late)
	}
	if late > 1 || late < 0.99 {
		t.Errorf("late rectification = %g, want just under 1", late)
	}
}
