package autograd

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// Optimizer updates a set of parameter nodes from the gradients a backward
// pass left on them. It is deliberately the same idea as the parent
// package's Optimizer, minus the network: here the parameters are just
// whichever nodes a model happens to be built from.
type Optimizer interface {
	// Step applies one update to every parameter.
	Step(params ...*Node)
}

// ZeroGrad clears the gradients of the given parameters, which a training
// loop does before each backward pass. Interior nodes need no clearing --
// Backward resets those itself -- so this is only ever called on leaves.
func ZeroGrad(params ...*Node) {
	for _, p := range params {
		p.Grad = matrix.New(p.Value.Rows, p.Value.Columns, nil)
	}
}

// SGDOptimizer is stochastic gradient descent with optional classical
// momentum: the step is a running average of past gradients rather than
// the latest one alone, which damps the zig-zag across a narrow valley and
// builds speed along it.
type SGDOptimizer struct {
	LearningRate float64
	Momentum     float64

	velocity map[*Node]matrix.Matrix
}

// SGD returns a plain gradient-descent optimizer; momentum of 0 disables
// the running average.
func SGD(learningRate, momentum float64) *SGDOptimizer {
	if learningRate <= 0 {
		panic("autograd: SGD needs a positive learning rate")
	}
	if momentum < 0 || momentum >= 1 {
		panic("autograd: SGD momentum must be in [0, 1)")
	}

	return &SGDOptimizer{
		LearningRate: learningRate,
		Momentum:     momentum,
		velocity:     map[*Node]matrix.Matrix{},
	}
}

// Step implements Optimizer.
func (o *SGDOptimizer) Step(params ...*Node) {
	for _, p := range params {
		step := p.Grad.Scale(o.LearningRate)

		if o.Momentum > 0 {
			v, ok := o.velocity[p]
			if !ok {
				v = matrix.New(p.Value.Rows, p.Value.Columns, nil)
			}
			v = v.Scale(o.Momentum).AddMatrix(step)
			o.velocity[p] = v
			step = v
		}

		p.Value = p.Value.SubtractMatrix(step)
	}
}

// AdamOptimizer is Adam (Kingma and Ba, 2014) over autograd parameters:
// per-parameter step sizes derived from running estimates of the
// gradient's mean and its variance, so a rarely-moving weight still gets a
// meaningful step and a wildly-swinging one is damped.
type AdamOptimizer struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64

	m map[*Node]matrix.Matrix
	v map[*Node]matrix.Matrix
	t int
}

// Adam returns an Adam optimizer with the paper's default coefficients.
func Adam(learningRate float64) *AdamOptimizer {
	if learningRate <= 0 {
		panic("autograd: Adam needs a positive learning rate")
	}

	return &AdamOptimizer{
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		m:            map[*Node]matrix.Matrix{},
		v:            map[*Node]matrix.Matrix{},
	}
}

// Step implements Optimizer.
func (o *AdamOptimizer) Step(params ...*Node) {
	o.t++

	// Both moment estimates start at zero and are therefore biased toward
	// it for the first several hundred steps; dividing by these corrections
	// undoes exactly that bias.
	correction1 := 1 - math.Pow(o.Beta1, float64(o.t))
	correction2 := 1 - math.Pow(o.Beta2, float64(o.t))

	for _, p := range params {
		m, ok := o.m[p]
		if !ok {
			m = matrix.New(p.Value.Rows, p.Value.Columns, nil)
		}
		v, ok := o.v[p]
		if !ok {
			v = matrix.New(p.Value.Rows, p.Value.Columns, nil)
		}

		m = m.Scale(o.Beta1).AddMatrix(p.Grad.Scale(1 - o.Beta1))
		v = v.Scale(o.Beta2).AddMatrix(p.Grad.HadamardProduct(p.Grad).Scale(1 - o.Beta2))
		o.m[p], o.v[p] = m, v

		lr, eps := o.LearningRate, o.Epsilon
		p.Value = p.Value.SubtractMatrix(m.Map(func(mean float64, i, j int) float64 {
			return lr * (mean / correction1) / (math.Sqrt(v.At(i, j)/correction2) + eps)
		}))
	}
}
