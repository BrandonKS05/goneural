package stepper

import (
	"errors"
	"math"
	"math/rand"
	"strings"

	"github.com/BrandonKS05/goneural/autograd"
	"github.com/BrandonKS05/goneural/matrix"
)

const (
	// MaxWordLength is the longest word a Stepper accepts.
	MaxWordLength = 8

	// MaxSampleLength caps how many letters Sample writes before giving up
	// on the model ever producing the end marker.
	MaxSampleLength = 12

	// Marker is the label of vocabulary entry 0, the token that stands for
	// both "before the first letter" and "after the last".
	Marker = "·"
)

// Stage names the part of a training step that was most recently
// completed.
type Stage int

const (
	// Ready means nothing has run yet: the weights are the initial ones.
	Ready Stage = iota
	// Forward has computed the hidden activations and output probabilities.
	Forward
	// Loss has scored those probabilities against the target letter.
	Loss
	// Backward has computed the gradient of that loss for every weight.
	Backward
	// Update has moved every weight against its gradient.
	Update
)

// String returns the stage's lower-case name.
func (s Stage) String() string {
	switch s {
	case Forward:
		return "forward"
	case Loss:
		return "loss"
	case Backward:
		return "backward"
	case Update:
		return "update"
	}
	return "ready"
}

// Pair is one training example: the id of a letter and the id of the
// letter that follows it.
type Pair struct {
	Input  int
	Target int
}

// Parameters is a plain copy of every learnable number in the model, or of
// the gradient of each.
type Parameters struct {
	// InputHidden[h][v] connects vocabulary entry v to hidden unit h.
	InputHidden [][]float64
	HiddenBias  []float64
	// HiddenOutput[v][h] connects hidden unit h to vocabulary entry v.
	HiddenOutput [][]float64
	OutputBias   []float64
}

// Layer names used by Change.
const (
	LayerInputHidden  = "inputHidden"
	LayerHiddenOutput = "hiddenOutput"
)

// Change records the connection weight an update moved furthest. From and
// To are unit indices in the direction signal flows: a vocabulary id and a
// hidden unit for LayerInputHidden, a hidden unit and a vocabulary id for
// LayerHiddenOutput. Biases are not considered; they are not connections.
type Change struct {
	Layer string
	From  int
	To    int
	Old   float64
	New   float64
}

// Snapshot is everything there is to see after a stage, copied out of the
// graph so it stays valid while training moves on.
type Snapshot struct {
	Word  string
	Vocab []string
	Pairs []Pair

	// Stage is the stage just completed, for the pair at index Pair.
	Stage Stage
	// Updates counts completed training steps.
	Updates int
	Pair    int
	Input   int
	Target  int

	// Hidden and Probabilities are nil until the current pair's forward
	// stage has run.
	Hidden        []float64
	Probabilities []float64

	// Loss is this pair's cross-entropy, meaningful once HasLoss is set.
	HasLoss bool
	Loss    float64

	Weights Parameters
	// Gradients are all zero until the current pair's backward stage.
	Gradients Parameters

	// LastChange describes the most recent update, nil before the first.
	LastChange *Change

	// AverageLoss is the mean loss over every pair with the current
	// weights; BestLoss is the lowest value it can possibly reach.
	AverageLoss float64
	BestLoss    float64
}

// Stepper trains a Model on the letter pairs of one word, a stage at a
// time. It is not safe for concurrent use.
type Stepper struct {
	word  string
	vocab []string
	pairs []Pair
	best  float64

	model     *Model
	params    []*autograd.Node
	optimizer autograd.Optimizer
	sampler   *rand.Rand

	stage   Stage
	pair    int
	updates int

	// The current pair's graph, kept between stages.
	hidden        *autograd.Node
	logits        *autograd.Node
	probabilities *autograd.Node
	loss          *autograd.Node

	lastChange *Change
}

// Vocabulary splits a word into the model's vocabulary and training pairs.
// Entry 0 is the marker; the letters follow in order of first appearance.
// The pairs are each character and its successor with the marker at both
// ends, so "hello" yields marker-h, h-e, e-l, l-l, l-o, o-marker.
//
// The word must be one to MaxWordLength lower-case letters a-z.
func Vocabulary(word string) (vocab []string, pairs []Pair, err error) {
	if word == "" {
		return nil, nil, errors.New("Type a word first.")
	}
	// Letters are checked before length, which counts bytes: a word like
	// "señorita" is eight letters but nine bytes, and should be refused for
	// its ñ rather than for being too long.
	for i := 0; i < len(word); i++ {
		if c := word[i]; c < 'a' || c > 'z' {
			return nil, nil, errors.New("Use only the letters a to z.")
		}
	}
	if len(word) > MaxWordLength {
		return nil, nil, errors.New("Keep it to 8 letters or fewer.")
	}

	ids := map[byte]int{}
	vocab = []string{Marker}
	sequence := []int{0}

	for i := 0; i < len(word); i++ {
		c := word[i]
		id, ok := ids[c]
		if !ok {
			id = len(vocab)
			ids[c] = id
			vocab = append(vocab, string(c))
		}
		sequence = append(sequence, id)
	}
	sequence = append(sequence, 0)

	for i := 0; i+1 < len(sequence); i++ {
		pairs = append(pairs, Pair{Input: sequence[i], Target: sequence[i+1]})
	}
	return vocab, pairs, nil
}

// BestLoss is the lowest average cross-entropy any model can reach on the
// pairs: the mean over the pairs of -ln(count(a->b) / count(a)). A model
// that sees only the current letter cannot do better than predict each
// successor in proportion to how often it follows, so the floor is above
// zero whenever a letter is followed by different letters.
func BestLoss(pairs []Pair) float64 {
	inputs := map[int]int{}
	transitions := map[Pair]int{}
	for _, p := range pairs {
		inputs[p.Input]++
		transitions[p]++
	}

	total := 0.0
	for _, p := range pairs {
		total -= math.Log(float64(transitions[p]) / float64(inputs[p.Input]))
	}
	return total / float64(len(pairs))
}

// New builds a Stepper for a word, with a freshly initialized model. The
// seed fixes the initial weights and the sampling sequence.
func New(word string, seed int64) (*Stepper, error) {
	vocab, pairs, err := Vocabulary(word)
	if err != nil {
		return nil, err
	}

	model := NewModel(len(vocab), seed)

	return &Stepper{
		word:      word,
		vocab:     vocab,
		pairs:     pairs,
		best:      BestLoss(pairs),
		model:     model,
		params:    model.Parameters(),
		optimizer: autograd.SGD(LearningRate, 0),
		// A separate stream, so drawing samples never disturbs anything a
		// seed is expected to reproduce.
		sampler: rand.New(rand.NewSource(seed ^ 0x5eed)),
		stage:   Ready,
	}, nil
}

// Model returns the model being trained.
func (s *Stepper) Model() *Model {
	return s.model
}

// Stage returns the stage most recently completed.
func (s *Stepper) Stage() Stage {
	return s.stage
}

// Step runs the next stage and returns the stage it ran. After Update the
// following call moves on to the next pair's Forward, wrapping around to
// the first pair after the last.
func (s *Stepper) Step() Stage {
	switch s.stage {
	case Ready, Update:
		if s.stage == Update {
			s.pair = (s.pair + 1) % len(s.pairs)
		}
		// The previous pair's gradients describe weights that have since
		// moved; clear them so a snapshot never shows stale blame.
		autograd.ZeroGrad(s.params...)
		s.loss = nil

		s.hidden, s.logits = s.model.Forward(s.pairs[s.pair].Input)
		// The loss stage fuses softmax into the cross-entropy and keeps
		// the probabilities to itself, so they are computed here for
		// display. This node is not part of the graph that is trained.
		s.probabilities = autograd.Softmax(s.logits)
		s.stage = Forward

	case Forward:
		s.loss = s.model.Loss(s.logits, s.pairs[s.pair].Target)
		s.stage = Loss

	case Loss:
		autograd.ZeroGrad(s.params...)
		s.loss.Backward()
		s.stage = Backward

	case Backward:
		inputHidden := s.model.Embedding.Table.Value
		hiddenOutput := s.model.Readout.Weight.Value

		s.optimizer.Step(s.params...)

		s.lastChange = biggestChange(inputHidden, hiddenOutput,
			s.model.Embedding.Table.Value, s.model.Readout.Weight.Value)
		s.updates++
		s.stage = Update
	}

	return s.stage
}

// StepPair runs stages until an Update completes: the rest of the current
// training step, or a whole one if none is in progress.
func (s *Stepper) StepPair() {
	for s.Step() != Update {
	}
}

// biggestChange finds the connection weight that differs most between the
// old and new matrices. The old values are safe to hold across the
// optimizer step because matrix operations never modify their receiver.
func biggestChange(oldInputHidden, oldHiddenOutput, newInputHidden, newHiddenOutput matrix.Matrix) *Change {
	change := &Change{Layer: LayerInputHidden}
	largest := -1.0

	for h := 0; h < oldInputHidden.Rows; h++ {
		for v := 0; v < oldInputHidden.Columns; v++ {
			before, after := oldInputHidden.At(h, v), newInputHidden.At(h, v)
			if d := math.Abs(after - before); d > largest {
				largest = d
				*change = Change{Layer: LayerInputHidden, From: v, To: h, Old: before, New: after}
			}
		}
	}
	for v := 0; v < oldHiddenOutput.Rows; v++ {
		for h := 0; h < oldHiddenOutput.Columns; h++ {
			before, after := oldHiddenOutput.At(v, h), newHiddenOutput.At(v, h)
			if d := math.Abs(after - before); d > largest {
				largest = d
				*change = Change{Layer: LayerHiddenOutput, From: h, To: v, Old: before, New: after}
			}
		}
	}

	return change
}

// AverageLoss is the mean loss over every pair with the current weights,
// computed as one batched forward pass. It builds its own graph and leaves
// the stage in progress untouched.
func (s *Stepper) AverageLoss() float64 {
	inputs := make([]int, len(s.pairs))
	targets := make([]int, len(s.pairs))
	for i, p := range s.pairs {
		inputs[i], targets[i] = p.Input, p.Target
	}

	_, logits := s.model.Forward(inputs...)
	return s.model.Loss(logits, targets...).Item()
}

// BestLoss is the floor AverageLoss can approach; see the function of the
// same name.
func (s *Stepper) BestLoss() float64 {
	return s.best
}

// Sample lets the model write a word: starting from the marker, it draws
// each next letter from the model's output distribution and feeds it back
// in, until the marker comes back or MaxSampleLength letters are written.
func (s *Stepper) Sample() string {
	var out strings.Builder

	id := 0
	for n := 0; n < MaxSampleLength; n++ {
		_, logits := s.model.Forward(id)
		probabilities := autograd.Softmax(logits).Value

		// Walk the cumulative distribution; the fallback to the last
		// entry covers the sum landing a rounding error short of 1.
		r := s.sampler.Float64()
		next := probabilities.Rows - 1
		for i := 0; i < probabilities.Rows; i++ {
			r -= probabilities.At(i, 0)
			if r < 0 {
				next = i
				break
			}
		}

		if next == 0 {
			break
		}
		out.WriteString(s.vocab[next])
		id = next
	}

	return out.String()
}

// Snapshot copies out the current state.
func (s *Stepper) Snapshot() Snapshot {
	pair := s.pairs[s.pair]

	snapshot := Snapshot{
		Word:    s.word,
		Vocab:   append([]string(nil), s.vocab...),
		Pairs:   append([]Pair(nil), s.pairs...),
		Stage:   s.stage,
		Updates: s.updates,
		Pair:    s.pair,
		Input:   pair.Input,
		Target:  pair.Target,
		Weights: Parameters{
			InputHidden:  rows(s.model.Embedding.Table.Value),
			HiddenBias:   s.model.HiddenBias.Value.Flatten(),
			HiddenOutput: rows(s.model.Readout.Weight.Value),
			OutputBias:   s.model.Readout.Bias.Value.Flatten(),
		},
		Gradients: Parameters{
			InputHidden:  rows(s.model.Embedding.Table.Grad),
			HiddenBias:   s.model.HiddenBias.Grad.Flatten(),
			HiddenOutput: rows(s.model.Readout.Weight.Grad),
			OutputBias:   s.model.Readout.Bias.Grad.Flatten(),
		},
		AverageLoss: s.AverageLoss(),
		BestLoss:    s.best,
	}

	if s.stage != Ready {
		snapshot.Hidden = s.hidden.Value.Flatten()
		snapshot.Probabilities = s.probabilities.Value.Flatten()
	}
	if s.loss != nil {
		snapshot.HasLoss = true
		snapshot.Loss = s.loss.Item()
	}
	if s.lastChange != nil {
		change := *s.lastChange
		snapshot.LastChange = &change
	}

	return snapshot
}

// rows copies a matrix into a slice per row.
func rows(m matrix.Matrix) [][]float64 {
	out := make([][]float64, m.Rows)
	for i := range out {
		out[i] = make([]float64, m.Columns)
		for j := range out[i] {
			out[i][j] = m.At(i, j)
		}
	}
	return out
}
