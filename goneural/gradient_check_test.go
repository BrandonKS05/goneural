package goneural

import (
	"math"
	"math/rand"
	"testing"
)

// TestMBGDMatchesComplexStepGradient cross-checks the analytic backprop
// gradient (MBGD) against the complex-step numeric gradient. Complex-step
// differentiation has no chain-rule to get wrong -- it just perturbs a
// weight by ih and reads the derivative out of the imaginary part of the
// forward pass -- so it acts as an independent oracle for what one descent
// step on a given weight *should* look like.
func TestMBGDMatchesComplexStepGradient(t *testing.T) {
	rand.Seed(42)

	n := New(0.1, MSE(),
		Layer{Nodes: 3},
		Layer{Nodes: 5, Activator: Sigmoid()},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 2, Activator: Sigmoid()},
	)

	input := []float64{0.5, -0.3, 0.8}
	target := []float64{1, 0}
	ds := DataSet{{Inputs: input, Targets: target}}

	const h = 1e-8
	const tol = 1e-4

	for l := range n.Weights {
		rows := n.Layers[l+1].Nodes
		cols := n.Layers[l].Nodes

		for _, rc := range [][2]int{{0, 0}, {rows - 1, cols - 1}} {
			r, c := rc[0], rc[1]

			trained := n.Copy()
			before := trained.Weights[l].Flatten()[r*cols+c]
			MBGD(1)(trained, ds)
			after := trained.Weights[l].Flatten()[r*cols+c]
			analyticStep := after - before

			yHat := complexForward(n, input, complexStepTarget{layer: l, row: r, col: c}, h)
			trueGrad := imag(complexMSE(yHat, target)) / h
			// MSE().FPrime omits the factor of 2 from d(x^2)/dx (a common
			// convention -- it's as if the loss were defined as half the
			// mean squared error), so the analytic gradient this library
			// actually computes is uniformly half the textbook MSE gradient.
			expectedStep := -n.LearningRate * trueGrad / 2

			if math.Abs(analyticStep-expectedStep) > tol {
				t.Errorf("layer %d weight[%d,%d]: backprop step = %g, complex-step-implied step = %g (diff %g)",
					l, r, c, analyticStep, expectedStep, math.Abs(analyticStep-expectedStep))
			}
		}
	}
}
