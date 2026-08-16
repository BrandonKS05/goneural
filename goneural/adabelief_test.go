package goneural

import (
	"math"
	"math/rand"
	"testing"
)

func TestAdaBeliefReducesLoss(t *testing.T) {
	rand.Seed(13)
	testOptimizerLearnsXOR(t, AdaBelief(2, 0.05), 500)
}

// TestAdaBeliefOutpacesAdamOnPredictableGradients is the whole thesis of the
// paper in one assertion. Fed the same perfectly consistent gradient, Adam
// divides by its magnitude and creeps; AdaBelief divides by the *surprise*,
// which collapses once momentum learns to predict the gradient, so it
// accelerates down exactly the slope Adam is most timid about.
func TestAdaBeliefOutpacesAdamOnPredictableGradients(t *testing.T) {
	const (
		lr    = 0.01
		beta1 = 0.9
		beta2 = 0.999
		eps   = 1e-8
		steps = 200
	)

	adamM, adamV := []float64{0}, []float64{0}
	beliefM, beliefS := []float64{0}, []float64{0}

	var adamStep, beliefStep float64
	for i := 0; i < steps; i++ {
		adamStep = adamWStep([]float64{1}, adamM, adamV, 1, lr, beta1, beta2, eps, 1, 1)[0]
		beliefStep = adaBeliefStep([]float64{1}, beliefM, beliefS, 1, lr, beta1, beta2, eps, 1, 1)[0]
	}

	if beliefStep <= adamStep {
		t.Fatalf("AdaBelief step %g did not exceed Adam step %g on a constant gradient", beliefStep, adamStep)
	}
}

// TestAdaBeliefDiscriminatesVarianceMoreSharplyThanAdam is the precise form of
// the claim. Both streams below have the same mean gradient of 1, so a plain
// SGD would treat them identically; they differ only in variance. Adam damps
// the noisy one somewhat, because E[g^2] absorbs both the mean and the
// variance. AdaBelief damps it far harder, because its accumulator is the
// variance alone -- the mean cancels out of (g - m). The ratio between how
// each optimizer treats the two streams is therefore much larger for
// AdaBelief, and that gap is the whole benefit of the method.
func TestAdaBeliefDiscriminatesVarianceMoreSharplyThanAdam(t *testing.T) {
	const (
		lr    = 0.01
		beta1 = 0.9
		beta2 = 0.999
		eps   = 1e-8
		steps = 4000 // beta2=0.999 has a ~1000 step memory; let it settle
	)

	// travel runs one gradient stream through one step function and returns
	// the mean absolute step over the second half, once the accumulators have
	// settled.
	travel := func(stream func(i int) float64, step func(g float64) float64) float64 {
		total := 0.0
		for i := 0; i < steps; i++ {
			s := step(stream(i))
			if i >= steps/2 {
				total += math.Abs(s)
			}
		}
		return total / float64(steps/2)
	}

	predictable := func(int) float64 { return 1 }
	noisy := func(i int) float64 {
		if i%2 == 0 {
			return 6 // mean is still 1, but the variance is large
		}
		return -4
	}

	adamStep := func() func(float64) float64 {
		m, v := []float64{0}, []float64{0}
		return func(g float64) float64 {
			return adamWStep([]float64{g}, m, v, 1, lr, beta1, beta2, eps, 1, 1)[0]
		}
	}
	beliefStep := func() func(float64) float64 {
		m, s := []float64{0}, []float64{0}
		return func(g float64) float64 {
			return adaBeliefStep([]float64{g}, m, s, 1, lr, beta1, beta2, eps, 1, 1)[0]
		}
	}

	adamRatio := travel(predictable, adamStep()) / travel(noisy, adamStep())
	beliefRatio := travel(predictable, beliefStep()) / travel(noisy, beliefStep())

	if beliefRatio <= adamRatio {
		t.Fatalf("AdaBelief predictable/noisy ratio %g did not exceed Adam's %g", beliefRatio, adamRatio)
	}

	// The separation should be dramatic, not marginal -- if this ever drops to
	// near parity the surprise term has stopped doing its job.
	if beliefRatio < 10*adamRatio {
		t.Errorf("AdaBelief ratio %g is only %.1fx Adam's %g, want at least 10x", beliefRatio, beliefRatio/adamRatio, adamRatio)
	}
}
