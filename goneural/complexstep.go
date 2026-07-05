package goneural

import (
	"math/cmplx"

	"github.com/BrandonKS05/goneural/matrix"
)

// complexStepTarget identifies the single scalar parameter that gets the
// imaginary perturbation ih during one complex-step forward pass.
type complexStepTarget struct {
	layer  int // index into n.Weights / n.Biases
	isBias bool
	row    int
	col    int // unused when isBias is true
}

// complexActivate evaluates the holomorphic (complex-analytic) continuation
// of a real activation function at a complex point. Sigmoid and identity are
// entire/holomorphic on the whole complex plane, so this is a mathematically
// exact extension, not an approximation.
func complexActivate(a Activation, z complex128) complex128 {
	if a.Name == sigmoid {
		return 1 / (1 + cmplx.Exp(-z))
	}
	return z // identity, and anything else validated upstream
}

// complexForward mirrors NeuralNetwork.predict but carries every value as
// complex128, adding an imaginary step h to exactly one weight or bias
// entry (target). The imaginary part of the resulting output encodes the
// sensitivity of the network to that one parameter.
func complexForward(n *NeuralNetwork, input []float64, target complexStepTarget, h float64) []complex128 {
	a := make([]complex128, len(input))
	for i, v := range input {
		a[i] = complex(v, 0)
	}

	for i := 0; i < len(n.Weights); i++ {
		rows := n.Layers[i+1].Nodes
		cols := n.Layers[i].Nodes

		wFlat := n.Weights[i].Flatten()
		bFlat := n.Biases[i].Flatten()

		next := make([]complex128, rows)
		for r := 0; r < rows; r++ {
			sum := complex(bFlat[r], 0)
			if target.isBias && target.layer == i && target.row == r {
				sum += complex(0, h)
			}

			for c := 0; c < cols; c++ {
				w := complex(wFlat[r*cols+c], 0)
				if !target.isBias && target.layer == i && target.row == r && target.col == c {
					w += complex(0, h)
				}
				sum += w * a[c]
			}

			next[r] = complexActivate(n.Layers[i+1].Activator, sum)
		}

		a = next
	}

	return a
}

// complexMSE is the same reduction as MSE().F (difference, square, sum,
// divide by row count) carried out over complex128 so the complex-step
// trick can read a derivative out of its imaginary part.
func complexMSE(yHat []complex128, y []float64) complex128 {
	sum := complex(0, 0)
	for i, target := range y {
		diff := yHat[i] - complex(target, 0)
		sum += diff * diff
	}
	return sum / complex(float64(len(y)), 0)
}

// validateComplexStep panics if the network uses anything the complex-step
// method can't differentiate correctly: a non-MSE loss, or an activation
// (ReLU, softmax) with no holomorphic extension. ReLU in particular is not
// complex-differentiable at all, so silently running it through would give
// a confidently wrong gradient instead of a loud error.
func validateComplexStep(n *NeuralNetwork) {
	if n.Loss.Name != mse {
		panic("goneural: complex-step differentiation currently only supports MSE loss")
	}
	for i := 1; i < len(n.Layers); i++ {
		name := n.Layers[i].Activator.Name
		if name != sigmoid && name != id {
			panic("goneural: complex-step differentiation only supports holomorphic activations (sigmoid, identity); " + string(name) + " has no holomorphic extension")
		}
	}
}

// ComplexStepGD is an experimental optimizer that estimates gradients with
// complex-step differentiation instead of backpropagation: each weight and
// bias is, one at a time, perturbed by a tiny imaginary step ih and run
// through the network, and Im(loss(w+ih))/h gives its partial derivative to
// (near) machine precision. Unlike real finite differences, this has no
// subtractive-cancellation error, since it never subtracts two nearly-equal
// real numbers to get a small one.
//
// It costs one full forward pass per scalar parameter per sample, so it's
// meant for small networks and experimentation, not as a faster substitute
// for backprop. h should be small (e.g. 1e-20 to 1e-6 all work equally well,
// since complex-step doesn't suffer the precision/truncation tradeoff a
// real-valued h does).
func ComplexStepGD(batchSize int, h float64) Optimizer {
	return func(n *NeuralNetwork, dataSet DataSet) float64 {
		validateComplexStep(n)
		err := 0.0

		for i := 0; i < len(dataSet); i += batchSize {
			batch := dataSet.Batch(i, batchSize)
			if len(batch) == 0 {
				continue
			}

			weightGrads := make([][]float64, len(n.Weights))
			biasGrads := make([][]float64, len(n.Biases))
			for l := range n.Weights {
				weightGrads[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
				biasGrads[l] = make([]float64, n.Biases[l].Rows)
			}

			for _, ds := range batch {
				err += n.Loss.F(n.predict(matrix.NewFromArray(ds.Inputs)), matrix.NewFromArray(ds.Targets))

				for l := range n.Weights {
					rows := n.Layers[l+1].Nodes
					cols := n.Layers[l].Nodes

					for r := 0; r < rows; r++ {
						for c := 0; c < cols; c++ {
							yHat := complexForward(n, ds.Inputs, complexStepTarget{layer: l, row: r, col: c}, h)
							weightGrads[l][r*cols+c] += imag(complexMSE(yHat, ds.Targets)) / h
						}

						yHat := complexForward(n, ds.Inputs, complexStepTarget{layer: l, isBias: true, row: r}, h)
						biasGrads[l][r] += imag(complexMSE(yHat, ds.Targets)) / h
					}
				}
			}

			lr := n.LearningRate / float64(len(batch))
			for l := range n.Weights {
				rows := n.Layers[l+1].Nodes
				cols := n.Layers[l].Nodes

				wDelta := matrix.Unflatten(rows, cols, scaled(weightGrads[l], lr))
				bDelta := matrix.Unflatten(rows, 1, scaled(biasGrads[l], lr))

				n.Weights[l] = n.Weights[l].SubtractMatrix(wDelta)
				n.Biases[l] = n.Biases[l].SubtractMatrix(bDelta)
			}
		}

		return err
	}
}

// ComplexStepSGD is ComplexStepGD with a batch size of 1, mirroring how SGD
// relates to MBGD.
func ComplexStepSGD(h float64) Optimizer {
	return ComplexStepGD(1, h)
}

func scaled(v []float64, s float64) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x * s
	}
	return out
}
