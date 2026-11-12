package stepper

import (
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/BrandonKS05/goneural/autograd"
)

func TestVocabularyAndPairs(t *testing.T) {
	vocab, pairs, err := Vocabulary("hello")
	if err != nil {
		t.Fatal(err)
	}

	if want := []string{Marker, "h", "e", "l", "o"}; !reflect.DeepEqual(vocab, want) {
		t.Errorf("vocab = %q, want %q", vocab, want)
	}

	// marker-h, h-e, e-l, l-l, l-o, o-marker
	want := []Pair{{0, 1}, {1, 2}, {2, 3}, {3, 3}, {3, 4}, {4, 0}}
	if !reflect.DeepEqual(pairs, want) {
		t.Errorf("pairs = %v, want %v", pairs, want)
	}
}

func TestVocabularyRejectsBadWords(t *testing.T) {
	for _, word := range []string{"", "Hello", "hi there", "café", "abcdefghi", "a1"} {
		if _, err := New(word, 1); err == nil {
			t.Errorf("New(%q) succeeded, want an error", word)
		}
	}
	for _, word := range []string{"a", "aa", "abcdefgh"} {
		if _, err := New(word, 1); err != nil {
			t.Errorf("New(%q) failed: %v", word, err)
		}
	}
}

// The page shows these messages as they are, so each bad word has to be
// refused for the reason a visitor would recognize.
func TestVocabularyErrorMessages(t *testing.T) {
	const (
		empty   = "Type a word first."
		letters = "Use only the letters a to z."
		long    = "Keep it to 8 letters or fewer."
	)

	cases := []struct {
		word string
		want string
	}{
		{"", empty},
		{"Hello", letters},
		{"hi there", letters},
		{"a1", letters},
		{"abcdefghi", long},
		{"abcdefghijklmnop", long},
		// Eight letters, but nine bytes: the ñ is the problem, not the length.
		{"señorita", letters},
		// Too long and not all letters: either answer is true, and the
		// letters come first.
		{"abcdefgh1", letters},
	}

	for _, c := range cases {
		_, _, err := Vocabulary(c.word)
		if err == nil {
			t.Errorf("Vocabulary(%q) succeeded, want %q", c.word, c.want)
			continue
		}
		if err.Error() != c.want {
			t.Errorf("Vocabulary(%q) = %q, want %q", c.word, err.Error(), c.want)
		}
	}
}

func TestInitialWeights(t *testing.T) {
	s, _ := New("hello", 3)
	snapshot := s.Snapshot()

	distinct := map[float64]bool{}
	for _, layer := range [][][]float64{snapshot.Weights.InputHidden, snapshot.Weights.HiddenOutput} {
		for _, row := range layer {
			for _, w := range row {
				if w < -InitRange || w > InitRange {
					t.Errorf("initial weight %g outside [-%g, %g]", w, InitRange, InitRange)
				}
				distinct[w] = true
			}
		}
	}
	if len(distinct) != 2*HiddenUnits*len(snapshot.Vocab) {
		t.Errorf("only %d distinct initial weights", len(distinct))
	}

	for _, b := range append(snapshot.Weights.HiddenBias, snapshot.Weights.OutputBias...) {
		if b != 0 {
			t.Errorf("initial bias %g, want 0", b)
		}
	}

	same, _ := New("hello", 3)
	if !reflect.DeepEqual(same.Snapshot().Weights, snapshot.Weights) {
		t.Error("the same seed produced different weights")
	}
	other, _ := New("hello", 4)
	if reflect.DeepEqual(other.Snapshot().Weights, snapshot.Weights) {
		t.Error("a different seed produced the same weights")
	}
}

// TestStagesMatchNormalTrainingStep is the claim the demo rests on: taking
// a training step apart into four stages does not change it. A second
// model from the same seed is trained with the loop every autograd program
// in this module uses, written out here with no reference to the stepper,
// and after every step the two must agree to the last bit.
func TestStagesMatchNormalTrainingStep(t *testing.T) {
	for _, word := range []string{"hello", "a", "banana", "abcdefgh"} {
		const seed = 42

		s, err := New(word, seed)
		if err != nil {
			t.Fatal(err)
		}

		_, pairs, _ := Vocabulary(word)
		reference := NewModel(len(s.vocab), seed)
		params := reference.Parameters()
		optimizer := autograd.SGD(LearningRate, 0)

		for step := 0; step < 5*len(pairs); step++ {
			pair := pairs[step%len(pairs)]

			_, logits := reference.Forward(pair.Input)
			loss := reference.Loss(logits, pair.Target)
			autograd.ZeroGrad(params...)
			loss.Backward()
			optimizer.Step(params...)

			for _, want := range []Stage{Forward, Loss, Backward, Update} {
				if got := s.Step(); got != want {
					t.Fatalf("%q step %d: ran stage %v, want %v", word, step, got, want)
				}
				// Reading state between stages must not disturb training.
				s.Snapshot()
				s.Sample()
			}

			snapshot := s.Snapshot()
			if snapshot.Loss != loss.Item() {
				t.Fatalf("%q step %d: loss %g, want %g", word, step, snapshot.Loss, loss.Item())
			}
			if snapshot.Input != pair.Input || snapshot.Target != pair.Target {
				t.Fatalf("%q step %d: trained on %d->%d, want %d->%d",
					word, step, snapshot.Input, snapshot.Target, pair.Input, pair.Target)
			}

			stepped := s.Model().Parameters()
			for i := range params {
				if !stepped[i].Value.Equal(params[i].Value) {
					t.Fatalf("%q step %d: parameter %s differs from the normal training step",
						word, step, params[i].Name)
				}
			}
		}
	}
}

func TestStepPairFinishesAStepFromAnyStage(t *testing.T) {
	whole, _ := New("hello", 9)
	for i := 0; i < 4; i++ {
		whole.StepPair()
	}

	mixed, _ := New("hello", 9)
	mixed.Step() // forward
	mixed.Step() // loss
	mixed.StepPair()
	mixed.StepPair()
	mixed.Step()
	mixed.StepPair()
	mixed.StepPair()

	if !reflect.DeepEqual(whole.Snapshot(), mixed.Snapshot()) {
		t.Error("StepPair from mid-step ended somewhere different from whole steps")
	}
	if got := whole.Snapshot(); got.Updates != 4 || got.Pair != 3 || got.Stage != Update {
		t.Errorf("after 4 steps: updates %d, pair %d, stage %v", got.Updates, got.Pair, got.Stage)
	}
}

// TestGradientsAgainstFiniteDifferences checks every gradient the backward
// stage reports against a central finite difference of the pair's loss:
// nudge one weight, rerun the forward pass, and see how the loss moved.
func TestGradientsAgainstFiniteDifferences(t *testing.T) {
	const (
		h   = 1e-6
		tol = 1e-6
	)

	s, _ := New("hello", 7)

	// Check every pair, across enough updates that the weights have left
	// their initial values.
	for step := 0; step < 3*len(s.pairs); step++ {
		s.Step()
		s.Step()
		if s.Step() != Backward {
			t.Fatal("expected the backward stage")
		}

		snapshot := s.Snapshot()
		pair := s.pairs[s.pair]

		lossAt := func() float64 {
			_, logits := s.model.Forward(pair.Input)
			return s.model.Loss(logits, pair.Target).Item()
		}

		reported := [][]float64{
			flatten(snapshot.Gradients.InputHidden),
			snapshot.Gradients.HiddenBias,
			flatten(snapshot.Gradients.HiddenOutput),
			snapshot.Gradients.OutputBias,
		}

		nonzero := 0
		for p, param := range s.params {
			original := param.Value
			for i := 0; i < original.Rows; i++ {
				for j := 0; j < original.Columns; j++ {
					bumped := func(offset float64) float64 {
						nudged := original.Copy()
						nudged.Set(i, j, original.At(i, j)+offset)
						param.Value = nudged
						defer func() { param.Value = original }()
						return lossAt()
					}

					numeric := (bumped(h) - bumped(-h)) / (2 * h)
					analytic := reported[p][i*original.Columns+j]

					if math.Abs(numeric-analytic) > tol*math.Max(1, math.Abs(numeric)) {
						t.Errorf("step %d: %s[%d][%d] gradient = %g, want the numeric %g",
							step, param.Name, i, j, analytic, numeric)
					}
					if analytic != 0 {
						nonzero++
					}
				}
			}
		}

		// Guard against the check passing vacuously on all-zero gradients:
		// one input column, both biases and every readout weight are live.
		if want := HiddenUnits + HiddenUnits + len(s.vocab)*HiddenUnits + len(s.vocab); nonzero != want {
			t.Errorf("step %d: %d nonzero gradients, want %d", step, nonzero, want)
		}

		s.Step() // update
	}
}

func flatten(m [][]float64) []float64 {
	var out []float64
	for _, row := range m {
		out = append(out, row...)
	}
	return out
}

func TestSnapshotFollowsTheStages(t *testing.T) {
	s, _ := New("hello", 5)

	ready := s.Snapshot()
	if ready.Stage != Ready || ready.Hidden != nil || ready.Probabilities != nil || ready.HasLoss || ready.LastChange != nil {
		t.Errorf("a fresh stepper should have nothing computed: %+v", ready)
	}

	s.Step()
	forward := s.Snapshot()
	if len(forward.Hidden) != HiddenUnits || len(forward.Probabilities) != len(forward.Vocab) || forward.HasLoss {
		t.Errorf("after forward: %+v", forward)
	}
	sum := 0.0
	for _, p := range forward.Probabilities {
		sum += p
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Errorf("probabilities sum to %g", sum)
	}

	s.Step()
	loss := s.Snapshot()
	if want := -math.Log(loss.Probabilities[loss.Target]); !loss.HasLoss || math.Abs(loss.Loss-want) > 1e-12 {
		t.Errorf("loss = %g, want -ln(p[target]) = %g", loss.Loss, want)
	}
	if sumAbs(flatten(loss.Gradients.HiddenOutput)) != 0 {
		t.Error("gradients should be zero before the backward stage")
	}

	s.Step()
	backward := s.Snapshot()
	if !reflect.DeepEqual(backward.Weights, loss.Weights) {
		t.Error("the backward stage moved the weights")
	}

	s.Step()
	update := s.Snapshot()
	change := update.LastChange
	if change == nil {
		t.Fatal("no change recorded after an update")
	}

	// Recompute the biggest move from the two snapshots and the gradients:
	// plain SGD moves each weight by exactly rate * gradient.
	largest := 0.0
	for hidden := range update.Weights.InputHidden {
		for v := range update.Weights.InputHidden[hidden] {
			before, after := backward.Weights.InputHidden[hidden][v], update.Weights.InputHidden[hidden][v]
			if want := before - LearningRate*backward.Gradients.InputHidden[hidden][v]; after != want {
				t.Errorf("input-hidden[%d][%d] = %g, want %g", hidden, v, after, want)
			}
			largest = math.Max(largest, math.Abs(after-before))
		}
	}
	for v := range update.Weights.HiddenOutput {
		for hidden := range update.Weights.HiddenOutput[v] {
			before, after := backward.Weights.HiddenOutput[v][hidden], update.Weights.HiddenOutput[v][hidden]
			largest = math.Max(largest, math.Abs(after-before))
		}
	}
	if got := math.Abs(change.New - change.Old); got != largest || largest == 0 {
		t.Errorf("recorded change moved %g, the largest move was %g", got, largest)
	}

	var before, after float64
	switch change.Layer {
	case LayerInputHidden:
		before, after = backward.Weights.InputHidden[change.To][change.From], update.Weights.InputHidden[change.To][change.From]
	case LayerHiddenOutput:
		before, after = backward.Weights.HiddenOutput[change.To][change.From], update.Weights.HiddenOutput[change.To][change.From]
	default:
		t.Fatalf("unknown layer %q", change.Layer)
	}
	if change.Old != before || change.New != after {
		t.Errorf("change says %g -> %g, the weights say %g -> %g", change.Old, change.New, before, after)
	}

	s.Step()
	next := s.Snapshot()
	if next.Pair != 1 || next.Stage != Forward || next.HasLoss || sumAbs(flatten(next.Gradients.HiddenOutput)) != 0 {
		t.Errorf("the next pair should start clean: pair %d, stage %v", next.Pair, next.Stage)
	}
}

func sumAbs(values []float64) float64 {
	total := 0.0
	for _, v := range values {
		total += math.Abs(v)
	}
	return total
}

func TestBestLoss(t *testing.T) {
	cases := []struct {
		word string
		want float64
	}{
		// Every letter has one successor: a perfect model is possible.
		{"cat", 0},
		// l is followed by l once and o once, so two of six pairs cost ln 2.
		{"hello", 2 * math.Ln2 / 6},
		// a is followed by a and by the marker.
		{"aa", 2 * math.Ln2 / 3},
	}

	for _, c := range cases {
		s, _ := New(c.word, 1)
		if got := s.BestLoss(); math.Abs(got-c.want) > 1e-12 {
			t.Errorf("BestLoss(%q) = %g, want %g", c.word, got, c.want)
		}
	}
}

func TestAverageLoss(t *testing.T) {
	s, _ := New("hello", 11)

	// Mid-step, with a graph in flight, to show the batched pass does not
	// interfere with it.
	s.Step()
	s.Step()

	total := 0.0
	for _, pair := range s.pairs {
		_, logits := s.model.Forward(pair.Input)
		total += s.model.Loss(logits, pair.Target).Item()
	}
	want := total / float64(len(s.pairs))

	if got := s.AverageLoss(); math.Abs(got-want) > 1e-12 {
		t.Errorf("AverageLoss = %g, want the mean of the per-pair losses %g", got, want)
	}
}

func TestTrainingApproachesBestLoss(t *testing.T) {
	for _, word := range []string{"hello", "cat", "banana"} {
		s, _ := New(word, 2)
		start := s.AverageLoss()

		for i := 0; i < 400; i++ {
			s.StepPair()
		}

		end, best := s.AverageLoss(), s.BestLoss()
		if end >= start {
			t.Errorf("%q: loss went from %g to %g", word, start, end)
		}
		if end < best-1e-9 {
			t.Errorf("%q: loss %g is below the floor %g", word, end, best)
		}
		if end > best+0.1 {
			t.Errorf("%q: loss %g is still far from the floor %g after 400 steps", word, end, best)
		}
	}
}

func TestSample(t *testing.T) {
	s, _ := New("hello", 6)

	for i := 0; i < 50; i++ {
		sample := s.Sample()
		if len(sample) > MaxSampleLength {
			t.Fatalf("sample %q is longer than %d letters", sample, MaxSampleLength)
		}
		if strings.Trim(sample, "helo") != "" {
			t.Fatalf("sample %q contains something outside the vocabulary", sample)
		}
	}

	// "cat" has no ambiguity, so a trained model should spell it nearly
	// every time.
	s, _ = New("cat", 6)
	for i := 0; i < 600; i++ {
		s.StepPair()
	}
	hits := 0
	for i := 0; i < 100; i++ {
		if s.Sample() == "cat" {
			hits++
		}
	}
	if hits < 90 {
		t.Errorf("a trained model wrote \"cat\" only %d times in 100", hits)
	}
}
