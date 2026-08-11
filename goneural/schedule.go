package goneural

import "math"

// Schedule maps a 1-based epoch number to a learning rate. Train invokes an
// Optimizer once per epoch, so a schedule composed with WithSchedule sees
// exactly one tick per epoch.
type Schedule func(epoch int) float64

// StepDecay multiplies the learning rate by factor once every `every`
// epochs: epochs 1..every run at initial, the next `every` at
// initial*factor, and so on. The classic "drop the LR 10x every N epochs"
// recipe is StepDecay(lr, 0.1, N).
func StepDecay(initial, factor float64, every int) Schedule {
	if every < 1 {
		panic("goneural: StepDecay needs a positive epoch interval")
	}

	return func(epoch int) float64 {
		if epoch < 1 {
			epoch = 1
		}
		return initial * math.Pow(factor, float64((epoch-1)/every))
	}
}

// ExponentialDecay shrinks the learning rate smoothly by e^(-decay) per
// epoch, starting from initial at epoch 1.
func ExponentialDecay(initial, decay float64) Schedule {
	return func(epoch int) float64 {
		if epoch < 1 {
			epoch = 1
		}
		return initial * math.Exp(-decay*float64(epoch-1))
	}
}

// CosineAnnealing sweeps the learning rate from initial down to min along a
// half cosine over each period of epochs, then restarts ("warm restarts",
// Loshchilov & Hutter, 2016). The restart jump back to the high rate helps
// training hop out of sharp minima that a monotonically decaying rate would
// settle into.
func CosineAnnealing(initial, min float64, period int) Schedule {
	if period < 1 {
		panic("goneural: CosineAnnealing needs a positive period")
	}

	return func(epoch int) float64 {
		if epoch < 1 {
			epoch = 1
		}
		phase := float64((epoch-1)%period) / float64(period)
		return min + 0.5*(initial-min)*(1+math.Cos(math.Pi*phase))
	}
}

// PolynomialDecay sweeps the learning rate from initial down to final over
// totalEpochs, following (1 - progress)^power: a power of 1 is a straight
// line, higher powers decay fast early then flatten, fractional powers do
// the reverse. From totalEpochs onward the rate holds at final.
func PolynomialDecay(initial, final float64, totalEpochs int, power float64) Schedule {
	if totalEpochs < 1 {
		panic("goneural: PolynomialDecay needs a positive epoch span")
	}
	if power <= 0 {
		panic("goneural: PolynomialDecay needs a positive power")
	}

	return func(epoch int) float64 {
		if epoch < 1 {
			epoch = 1
		}
		if epoch >= totalEpochs {
			return final
		}
		progress := float64(epoch-1) / float64(totalEpochs-1)
		return final + (initial-final)*math.Pow(1-progress, power)
	}
}

// WithWarmup prefixes a schedule with warmupEpochs of linear ramp: the
// rate climbs from base(1)/warmupEpochs up to base(1), then hands over to
// the base schedule, which starts fresh from its own epoch 1. Warming up
// keeps the very first updates small while adaptive optimizers' statistics
// (and momentum) are still noise, which is when a full-size learning rate
// does the most damage.
func WithWarmup(base Schedule, warmupEpochs int) Schedule {
	if warmupEpochs < 1 {
		panic("goneural: WithWarmup needs a positive warmup span")
	}

	return func(epoch int) float64 {
		if epoch < 1 {
			epoch = 1
		}
		if epoch <= warmupEpochs {
			return base(1) * float64(epoch) / float64(warmupEpochs)
		}
		return base(epoch - warmupEpochs)
	}
}

// WithSchedule wraps an optimizer so its k-th invocation runs with the
// network's LearningRate set to schedule(k). This drives the optimizers
// that read n.LearningRate (SGD, MBGD, GD, Adam); for the stateful
// optimizers that carry their own LearningRate field, use WithScheduleFunc
// and point apply at that field instead.
func WithSchedule(optimizer Optimizer, schedule Schedule) Optimizer {
	epoch := 0
	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		epoch++
		n.LearningRate = schedule(epoch)
		return optimizer(n, dataSet)
	}
}

// WithScheduleFunc wraps an optimizer so that before its k-th invocation,
// apply is called with schedule(k). It exists for optimizers whose learning
// rate lives outside the network, e.g.:
//
//	o := NewMomentumOptimizer(2, 0.5, 0.9)
//	opt := WithScheduleFunc(o.Optimize, CosineAnnealing(0.5, 0.01, 50),
//		func(lr float64) { o.LearningRate = lr })
func WithScheduleFunc(optimizer Optimizer, schedule Schedule, apply func(lr float64)) Optimizer {
	epoch := 0
	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		epoch++
		apply(schedule(epoch))
		return optimizer(n, dataSet)
	}
}
