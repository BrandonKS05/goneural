// Package stepper drives a tiny next-letter language model through one
// training step a stage at a time -- forward, loss, backward, update -- so
// that something outside the program (a web page, in practice) can stop
// between the stages and look at every number.
//
// Nothing here computes a derivative or an activation of its own. The
// model is assembled from the autograd package's layers and operations,
// and the four stages are the four calls any autograd training loop
// already makes, merely not made back to back.
package stepper

import (
	"math/rand"

	"github.com/BrandonKS05/goneural/autograd"
	"github.com/BrandonKS05/goneural/matrix"
)

const (
	// HiddenUnits is the width of the single tanh layer.
	HiddenUnits = 4

	// LearningRate is the plain SGD step size.
	LearningRate = 0.5

	// InitRange bounds the initial weights: uniform in [-InitRange, InitRange].
	InitRange = 0.8
)

// Model is the whole network: a letter id looks up a column of the
// embedding table (which is exactly a one-hot vector times a weight
// matrix), a bias and tanh make the hidden layer, and a dense readout
// produces one logit per vocabulary entry.
type Model struct {
	// Embedding holds the input-to-hidden weights, HiddenUnits x vocab.
	Embedding *autograd.Embedding

	// HiddenBias is the hidden layer's bias, HiddenUnits x 1.
	HiddenBias *autograd.Node

	// Readout holds the hidden-to-output weights (vocab x HiddenUnits) and
	// the output bias.
	Readout *autograd.Linear
}

// NewModel builds a model over a vocabulary of the given size, with
// weights drawn uniformly from [-InitRange, InitRange] and zero biases.
// The same seed always produces the same model.
//
// The layers are assembled from their exported fields rather than through
// NewEmbedding and NewLinear, because those draw from the global generator
// with initializations of their own choosing; this model needs a specific
// range and a seed it can be rebuilt from.
func NewModel(vocab int, seed int64) *Model {
	if vocab < 2 {
		panic("stepper: a model needs at least one letter and the marker")
	}

	rng := rand.New(rand.NewSource(seed))
	uniform := func(rows, columns int) matrix.Matrix {
		return matrix.New(rows, columns, nil).Map(func(_ float64, _, _ int) float64 {
			return (rng.Float64()*2 - 1) * InitRange
		})
	}

	return &Model{
		Embedding: &autograd.Embedding{
			Table: autograd.Param(uniform(HiddenUnits, vocab)).Named("input-hidden.weight"),
		},
		HiddenBias: autograd.Param(matrix.New(HiddenUnits, 1, nil)).Named("hidden.bias"),
		Readout: &autograd.Linear{
			Weight: autograd.Param(uniform(vocab, HiddenUnits)).Named("hidden-output.weight"),
			Bias:   autograd.Param(matrix.New(vocab, 1, nil)).Named("output.bias"),
		},
	}
}

// Vocab returns the vocabulary size the model was built for.
func (m *Model) Vocab() int {
	return m.Embedding.Table.Value.Columns
}

// Parameters returns the learnable nodes, in a fixed order: input-hidden
// weights, hidden bias, hidden-output weights, output bias.
func (m *Model) Parameters() []*autograd.Node {
	return []*autograd.Node{m.Embedding.Table, m.HiddenBias, m.Readout.Weight, m.Readout.Bias}
}

// Forward runs the given letter ids through the network, one column per
// id, and returns the hidden activations and the output logits.
func (m *Model) Forward(ids ...int) (hidden, logits *autograd.Node) {
	hidden = autograd.Tanh(autograd.AddBias(m.Embedding.Forward(ids), m.HiddenBias))
	return hidden, m.Readout.Forward(hidden)
}

// Loss is the mean cross-entropy between the logits' softmax and the
// target ids, one target per column of logits.
func (m *Model) Loss(logits *autograd.Node, targets ...int) *autograd.Node {
	oneHot := matrix.New(m.Vocab(), len(targets), nil)
	for j, id := range targets {
		oneHot.Set(id, j, 1)
	}
	return autograd.SoftmaxCrossEntropy(logits, autograd.Const(oneHot))
}
