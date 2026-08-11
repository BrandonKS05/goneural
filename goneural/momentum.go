package goneural

import (
	"github.com/BrandonKS05/goneural/matrix"
)

// MomentumOptimizer is mini-batch gradient descent with momentum: every
// parameter accumulates an exponentially decaying velocity of its past
// gradients and steps along that velocity instead of the raw gradient
// (Polyak's heavy-ball method). The running average smooths out noisy
// per-batch gradients and builds speed along directions that stay
// consistent from batch to batch, which usually gets through flat regions
// and shallow ravines much faster than plain MBGD.
//
// With Nesterov enabled (Sutskever et al., 2013), the current gradient is
// folded into the decayed velocity once more before stepping, so the update
// reflects where the velocity is headed rather than where it was -- this
// "look ahead" corrects overshoot one update earlier than classic momentum.
type MomentumOptimizer struct {
	BatchSize    int
	LearningRate float64
	Momentum     float64
	Nesterov     bool

	// MaxGradNorm, when positive, clips each batch's averaged gradients
	// (weights and biases together) by global norm before they enter the
	// velocity, so one pathological batch can't fling the velocity off --
	// see ClipByGlobalNorm. Zero leaves gradients unclipped.
	MaxGradNorm float64

	velocityW []matrix.Matrix
	velocityB []matrix.Matrix
}

// NewMomentumOptimizer creates a momentum optimizer. A Momentum of 0.9 is
// the usual starting point; 0 degrades to plain mini-batch gradient descent.
func NewMomentumOptimizer(batchSize int, learningRate, momentum float64) *MomentumOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &MomentumOptimizer{
		BatchSize:    batchSize,
		LearningRate: learningRate,
		Momentum:     momentum,
	}
}

// MomentumSGD returns an Optimizer that uses classic (heavy-ball) momentum.
func MomentumSGD(batchSize int, learningRate, momentum float64) Optimizer {
	return NewMomentumOptimizer(batchSize, learningRate, momentum).Optimize
}

// NesterovSGD returns an Optimizer that uses Nesterov accelerated momentum.
func NesterovSGD(batchSize int, learningRate, momentum float64) Optimizer {
	o := NewMomentumOptimizer(batchSize, learningRate, momentum)
	o.Nesterov = true
	return o.Optimize
}

// Optimize implements the Optimizer signature.
func (o *MomentumOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	lenWeights := len(n.Weights)
	if o.velocityW == nil {
		o.velocityW = make([]matrix.Matrix, lenWeights)
		o.velocityB = make([]matrix.Matrix, lenWeights)
		for l := 0; l < lenWeights; l++ {
			o.velocityW[l] = matrix.New(n.Weights[l].Rows, n.Weights[l].Columns, nil)
			o.velocityB[l] = matrix.New(n.Biases[l].Rows, n.Biases[l].Columns, nil)
		}
	}

	err := 0.0

	for i := 0; i < len(dataSet); i += o.BatchSize {
		batch := dataSet.Batch(i, o.BatchSize)
		if len(batch) == 0 {
			continue
		}

		weightGrads := make([]matrix.Matrix, lenWeights)
		biasGrads := make([]matrix.Matrix, lenWeights)
		for l := 0; l < lenWeights; l++ {
			weightGrads[l] = matrix.New(n.Weights[l].Rows, n.Weights[l].Columns, nil)
			biasGrads[l] = matrix.New(n.Biases[l].Rows, n.Biases[l].Columns, nil)
		}

		for _, ds := range batch {
			outputs := n.predict(matrix.NewFromArray(ds.Inputs))
			targets := matrix.NewFromArray(ds.Targets)
			err += n.Loss.F(outputs, targets)

			// Same backprop recurrence as MBGD, but like Adam this needs the
			// true (positive) gradient dLoss/dw, since the velocity update
			// works out the descent direction itself.
			delta := outputDelta(n, outputs, targets).Scale(-1)
			for l := lenWeights - 1; l >= 0; l-- {
				weightGrads[l] = weightGrads[l].AddMatrix(delta.Multiply(n.Activations[l].Transpose()))
				biasGrads[l] = biasGrads[l].AddMatrix(delta)

				if l > 0 {
					delta = n.Weights[l].Transpose().Multiply(delta)
					delta = n.Activations[l].
						Map(func(val float64, x, y int) float64 {
							return n.Layers[l].Activator.FPrime(val)
						}).
						HadamardProduct(delta)
				}
			}
		}

		avgW := make([]matrix.Matrix, lenWeights)
		avgB := make([]matrix.Matrix, lenWeights)
		for l := 0; l < lenWeights; l++ {
			avgW[l] = weightGrads[l].Divide(float64(len(batch)))
			avgB[l] = biasGrads[l].Divide(float64(len(batch)))
		}

		if o.MaxGradNorm > 0 {
			// Clip weights and biases as one collection so the shared
			// rescale factor keeps the overall gradient direction intact.
			clipped := ClipByGlobalNorm(append(append([]matrix.Matrix{}, avgW...), avgB...), o.MaxGradNorm)
			avgW = clipped[:lenWeights]
			avgB = clipped[lenWeights:]
		}

		for l := 0; l < lenWeights; l++ {
			wGrad := avgW[l]
			bGrad := avgB[l]

			o.velocityW[l] = o.velocityW[l].Scale(o.Momentum).AddMatrix(wGrad)
			o.velocityB[l] = o.velocityB[l].Scale(o.Momentum).AddMatrix(bGrad)

			wStep := o.velocityW[l]
			bStep := o.velocityB[l]
			if o.Nesterov {
				// Look ahead: fold the current gradient into the decayed
				// velocity once more, stepping toward where the velocity is
				// going instead of where it has been.
				wStep = wGrad.AddMatrix(o.velocityW[l].Scale(o.Momentum))
				bStep = bGrad.AddMatrix(o.velocityB[l].Scale(o.Momentum))
			}

			n.Weights[l] = n.Weights[l].SubtractMatrix(wStep.Scale(o.LearningRate))
			n.Biases[l] = n.Biases[l].SubtractMatrix(bStep.Scale(o.LearningRate))
		}
	}

	return err
}
