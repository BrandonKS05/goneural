package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// AdaptiveOptimizer is an advanced optimizer that extends Adam with optional
// decoupled weight decay and a reusable optimizer object that can keep state
// across epochs.
type AdaptiveOptimizer struct {
	BatchSize    int
	LearningRate float64
	WeightDecay  float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64

	mW [][]float64
	vW [][]float64
	mB [][]float64
	vB [][]float64
	t  int
}

// NewAdaptiveOptimizer creates an adaptive optimizer with the given batch size,
// learning rate, and optional weight decay.
func NewAdaptiveOptimizer(batchSize int, learningRate, weightDecay float64) *AdaptiveOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &AdaptiveOptimizer{
		BatchSize:    batchSize,
		LearningRate: learningRate,
		WeightDecay:  weightDecay,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
	}
}

// AdamW returns an Optimizer that uses Adam with decoupled weight decay.
func AdamW(batchSize int, learningRate, weightDecay float64) Optimizer {
	return NewAdaptiveOptimizer(batchSize, learningRate, weightDecay).Optimize
}

// Optimize implements the Optimizer signature.
func (o *AdaptiveOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	lenWeights := len(n.Weights)
	if o.mW == nil {
		o.mW = make([][]float64, lenWeights)
		o.vW = make([][]float64, lenWeights)
		o.mB = make([][]float64, lenWeights)
		o.vB = make([][]float64, lenWeights)
		for l := 0; l < lenWeights; l++ {
			o.mW[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
			o.vW[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
			o.mB[l] = make([]float64, n.Biases[l].Rows)
			o.vB[l] = make([]float64, n.Biases[l].Rows)
		}
	}

	err := 0.0

	for i := 0; i < len(dataSet); i += o.BatchSize {
		batch := dataSet.Batch(i, o.BatchSize)
		if len(batch) == 0 {
			continue
		}

		weightGrads, biasGrads, batchErr := accumulateBatchGradients(n, batch)
		err += batchErr

		o.t++
		biasCorrection1 := 1 - math.Pow(o.Beta1, float64(o.t))
		biasCorrection2 := 1 - math.Pow(o.Beta2, float64(o.t))

		for l := 0; l < lenWeights; l++ {
			rows := n.Layers[l+1].Nodes
			cols := n.Layers[l].Nodes

			wStep := adamWStep(weightGrads[l].Flatten(), o.mW[l], o.vW[l], len(batch), o.LearningRate, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1, biasCorrection2)
			bStep := adamWStep(biasGrads[l].Flatten(), o.mB[l], o.vB[l], len(batch), o.LearningRate, o.Beta1, o.Beta2, o.Epsilon, biasCorrection1, biasCorrection2)

			weightDecayStep := matrix.New(n.Weights[l].Rows, n.Weights[l].Columns, nil)
			if o.WeightDecay != 0 {
				weightDecayStep = n.Weights[l].Scale(o.LearningRate * o.WeightDecay)
			}

			n.Weights[l] = n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, wStep)).SubtractMatrix(weightDecayStep)
			n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, bStep))
		}
	}

	return err
}

func adamWStep(gradSum, m, v []float64, batchLen int, lr, beta1, beta2, epsilon, biasCorrection1, biasCorrection2 float64) []float64 {
	step := make([]float64, len(gradSum))
	for k, g := range gradSum {
		g /= float64(batchLen)
		m[k] = beta1*m[k] + (1-beta1)*g
		v[k] = beta2*v[k] + (1-beta2)*g*g
		mHat := m[k] / biasCorrection1
		vHat := v[k] / biasCorrection2
		step[k] = lr * mHat / (math.Sqrt(vHat) + epsilon)
	}
	return step
}

// RMSPropOptimizer is an optimizer that uses root-mean-square scaling of
// gradients to adapt learning rates per parameter.
type RMSPropOptimizer struct {
	BatchSize    int
	LearningRate float64
	Decay        float64
	Epsilon      float64
	Beta         float64

	meanSquares [][]float64
	biasSquares [][]float64
}

// NewRMSPropOptimizer creates a new RMSProp optimizer with default decay.
func NewRMSPropOptimizer(batchSize int, learningRate float64) *RMSPropOptimizer {
	if batchSize < 1 {
		batchSize = 1
	}

	return &RMSPropOptimizer{
		BatchSize:    batchSize,
		LearningRate: learningRate,
		Decay:        0.9,
		Epsilon:      1e-8,
		Beta:         0.9,
	}
}

// RMSProp returns an Optimizer that uses RMSProp.
func RMSProp(batchSize int, learningRate float64) Optimizer {
	return NewRMSPropOptimizer(batchSize, learningRate).Optimize
}

// Optimize implements the Optimizer signature for RMSProp.
func (o *RMSPropOptimizer) Optimize(n *NeuralNetwork, dataSet DataSet) float64 {
	lenWeights := len(n.Weights)
	if o.meanSquares == nil {
		o.meanSquares = make([][]float64, lenWeights)
		o.biasSquares = make([][]float64, lenWeights)
		for l := 0; l < lenWeights; l++ {
			o.meanSquares[l] = make([]float64, n.Weights[l].Rows*n.Weights[l].Columns)
			o.biasSquares[l] = make([]float64, n.Biases[l].Rows)
		}
	}

	err := 0.0

	for i := 0; i < len(dataSet); i += o.BatchSize {
		batch := dataSet.Batch(i, o.BatchSize)
		if len(batch) == 0 {
			continue
		}

		weightGrads, biasGrads, batchErr := accumulateBatchGradients(n, batch)
		err += batchErr

		for l := 0; l < lenWeights; l++ {
			rows := n.Layers[l+1].Nodes
			cols := n.Layers[l].Nodes

			weightFlat := weightGrads[l].Flatten()
			biasFlat := biasGrads[l].Flatten()

			for k := range weightFlat {
				weightFlat[k] /= float64(len(batch))
				o.meanSquares[l][k] = o.Decay*o.meanSquares[l][k] + (1-o.Decay)*weightFlat[k]*weightFlat[k]
			}

			for k := range biasFlat {
				biasFlat[k] /= float64(len(batch))
				o.biasSquares[l][k] = o.Decay*o.biasSquares[l][k] + (1-o.Decay)*biasFlat[k]*biasFlat[k]
			}

			weightStep := make([]float64, len(weightFlat))
			for k, g := range weightFlat {
				weightStep[k] = o.LearningRate * g / (math.Sqrt(o.meanSquares[l][k]) + o.Epsilon)
			}

			biasStep := make([]float64, len(biasFlat))
			for k, g := range biasFlat {
				biasStep[k] = o.LearningRate * g / (math.Sqrt(o.biasSquares[l][k]) + o.Epsilon)
			}

			n.Weights[l] = n.Weights[l].SubtractMatrix(matrix.Unflatten(rows, cols, weightStep))
			n.Biases[l] = n.Biases[l].SubtractMatrix(matrix.Unflatten(rows, 1, biasStep))
		}
	}

	return err
}

// Trainer provides a higher-level training loop with optional early stopping.
type Trainer struct {
	Network       *NeuralNetwork
	Optimizer     Optimizer
	Epochs        int
	EarlyStopping bool
	Patience      int
	MinDelta      float64
}

// NewTrainer creates a Trainer for the given network and optimizer.
func NewTrainer(network *NeuralNetwork, optimizer Optimizer, epochs int) *Trainer {
	return &Trainer{
		Network:   network,
		Optimizer: optimizer,
		Epochs:    epochs,
		Patience:  10,
		MinDelta:  1e-8,
	}
}

// Train runs the configured optimizer over the dataset for up to Epochs.
// If EarlyStopping is enabled, training stops when the loss does not improve
// for Patience epochs.
func (t *Trainer) Train(dataSet DataSet) float64 {
	if len(dataSet) == 0 {
		return 0
	}

	bestErr := math.Inf(1)
	noImprovement := 0
	bestWeights := make([]matrix.Matrix, len(t.Network.Weights))
	bestBiases := make([]matrix.Matrix, len(t.Network.Biases))

	for l := range t.Network.Weights {
		bestWeights[l] = t.Network.Weights[l].Copy()
		bestBiases[l] = t.Network.Biases[l].Copy()
	}

	for epoch := 1; epoch <= t.Epochs; epoch++ {
		err := t.Optimizer(t.Network, dataSet)
		avgErr := err / float64(len(dataSet))

		if avgErr < bestErr-t.MinDelta {
			bestErr = avgErr
			noImprovement = 0
			for l := range t.Network.Weights {
				bestWeights[l] = t.Network.Weights[l].Copy()
				bestBiases[l] = t.Network.Biases[l].Copy()
			}
		} else {
			noImprovement++
		}

		if t.EarlyStopping && t.Patience > 0 && noImprovement >= t.Patience {
			break
		}
	}

	for l := range t.Network.Weights {
		t.Network.Weights[l] = bestWeights[l]
		t.Network.Biases[l] = bestBiases[l]
	}

	return bestErr
}
