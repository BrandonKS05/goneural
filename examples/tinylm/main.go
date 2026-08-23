// Command tinylm trains a small character-level transformer language model
// and then generates text with it.
//
// The model is the real arrangement, just scaled down: characters become
// embeddings, positional encoding is added, a stack of pre-norm
// transformer blocks with causal masking processes the sequence, and a
// final projection predicts the next character at every position at once.
// Training is next-character prediction on a tiny corpus; generation feeds
// the model its own output one character at a time.
//
// Everything is built from the autograd package. No gradient anywhere in
// this program, or in the layers it uses, is written by hand.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/BrandonKS05/goneural/autograd"
	"github.com/BrandonKS05/goneural/matrix"
)

const (
	modelWidth = 64
	heads      = 4
	blocks     = 2
	context    = 32
	steps      = 3000
	learnRate  = 0.003
)

// corpus is deliberately small and highly structured, so a model this size
// can actually learn it: the interesting question is whether it picks up
// the structure, not whether it memorizes English.
const corpus = "the cat sat on the mat. the dog sat on the log. " +
	"the cat ran to the dog. the dog ran to the mat. " +
	"a cat and a dog sat on a log. a dog and a cat ran to a mat. "

// vocabulary maps between characters and the ids the model works in.
type vocabulary struct {
	runes []rune
	ids   map[rune]int
}

func newVocabulary(text string) *vocabulary {
	seen := map[rune]bool{}
	for _, r := range text {
		seen[r] = true
	}

	v := &vocabulary{ids: map[rune]int{}}
	for r := range seen {
		v.runes = append(v.runes, r)
	}
	sort.Slice(v.runes, func(i, j int) bool { return v.runes[i] < v.runes[j] })

	for i, r := range v.runes {
		v.ids[r] = i
	}
	return v
}

func (v *vocabulary) encode(text string) []int {
	out := make([]int, 0, len(text))
	for _, r := range text {
		out = append(out, v.ids[r])
	}
	return out
}

func (v *vocabulary) decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		b.WriteRune(v.runes[id])
	}
	return b.String()
}

// model is a stack of transformer blocks between an embedding table and a
// projection back to vocabulary logits.
type model struct {
	embedding *autograd.Embedding
	blocks    []*autograd.TransformerBlock
	final     *autograd.LayerNorm
	readout   *autograd.Linear
	vocab     *vocabulary
}

func newModel(v *vocabulary) *model {
	m := &model{
		embedding: autograd.NewEmbedding(len(v.runes), modelWidth),
		final:     autograd.NewLayerNorm(modelWidth),
		readout:   autograd.NewLinear(modelWidth, len(v.runes)),
		vocab:     v,
	}
	for i := 0; i < blocks; i++ {
		m.blocks = append(m.blocks, autograd.NewTransformerBlock(modelWidth, heads))
	}
	return m
}

func (m *model) parameters() []*autograd.Node {
	params := m.embedding.Parameters()
	for _, b := range m.blocks {
		params = append(params, b.Parameters()...)
	}
	params = append(params, m.final.Parameters()...)
	return append(params, m.readout.Parameters()...)
}

// forward maps a sequence of ids to vocab x positions logits: column j is
// the model's prediction for what follows position j.
func (m *model) forward(ids []int, training bool) *autograd.Node {
	x := autograd.Add(m.embedding.Forward(ids), autograd.PositionalEncoding(modelWidth, len(ids)))

	// The causal mask is what makes training on every position at once
	// legitimate: without it, predicting position j could just read
	// position j+1.
	mask := autograd.CausalMask(len(ids))
	for _, b := range m.blocks {
		x = b.Forward(x, mask, training)
	}

	return m.readout.Forward(m.final.Forward(x))
}

// oneHot builds the target matrix for a window: the next character at
// every position.
func oneHot(vocab int, ids []int) matrix.Matrix {
	m := matrix.New(vocab, len(ids), nil)
	for j, id := range ids {
		m.Set(id, j, 1)
	}
	return m
}

func main() {
	rand.Seed(7)

	vocab := newVocabulary(corpus)
	data := vocab.encode(corpus)
	m := newModel(vocab)
	params := m.parameters()

	count := 0
	for _, p := range params {
		count += p.Value.Rows * p.Value.Columns
	}
	fmt.Printf("character-level transformer: %d blocks, %d heads, width %d, %d parameters\n",
		blocks, heads, modelWidth, count)
	fmt.Printf("corpus of %d characters, vocabulary of %d\n\n", len(data), len(vocab.runes))

	optimizer := autograd.Adam(learnRate)
	running := 0.0

	for step := 1; step <= steps; step++ {
		// A random window, with the targets being the same window shifted
		// one character to the left.
		start := rand.Intn(len(data) - context - 1)
		inputs := data[start : start+context]
		targets := data[start+1 : start+context+1]

		logits := m.forward(inputs, true)
		loss := autograd.SoftmaxCrossEntropy(logits,
			autograd.Const(oneHot(len(vocab.runes), targets)))

		autograd.ZeroGrad(params...)
		loss.Backward()
		optimizer.Step(params...)

		running += loss.Item()
		if step%300 == 0 {
			mean := running / 300
			// Perplexity is the loss exponentiated: how many characters the
			// model is effectively choosing between at each step. It starts
			// at the vocabulary size and should fall toward 1.
			fmt.Printf("  step %4d   loss %.4f   perplexity %5.2f\n", step, mean, math.Exp(mean))
			running = 0
		}
	}

	fmt.Printf("\n(an untrained model's perplexity would be about %d, the vocabulary size)\n", len(vocab.runes))

	fmt.Println("\ngenerated continuations, sampled at three temperatures")
	for _, temperature := range []float64{0.2, 0.6, 1.0} {
		fmt.Printf("\n  T=%.1f  %q\n", temperature, m.generate("the cat", 60, temperature))
	}

	// What one head looks at while reading a prompt. Attention weights are
	// a distribution, so this is the model's own account of where each
	// prediction came from.
	fmt.Println("\nhead 0 of block 0, reading \"the cat sat on\":")
	m.showAttention("the cat sat on")
}

// generate continues a prompt one character at a time, feeding each
// sampled character back in.
func (m *model) generate(prompt string, length int, temperature float64) string {
	ids := m.vocab.encode(prompt)

	for i := 0; i < length; i++ {
		// Only the most recent context characters are visible, which is the
		// model's whole window.
		window := ids
		if len(window) > context {
			window = window[len(window)-context:]
		}

		logits := m.forward(window, false).Value
		ids = append(ids, sample(logits, len(window)-1, temperature))
	}

	return m.vocab.decode(ids)
}

// sample draws a character from the distribution at one position, with
// temperature sharpening (below 1) or flattening (above 1) the choice.
func sample(logits matrix.Matrix, position int, temperature float64) int {
	max := math.Inf(-1)
	for i := 0; i < logits.Rows; i++ {
		max = math.Max(max, logits.At(i, position)/temperature)
	}

	probabilities := make([]float64, logits.Rows)
	sum := 0.0
	for i := range probabilities {
		probabilities[i] = math.Exp(logits.At(i, position)/temperature - max)
		sum += probabilities[i]
	}

	draw := rand.Float64() * sum
	for i, p := range probabilities {
		draw -= p
		if draw <= 0 {
			return i
		}
	}
	return len(probabilities) - 1
}

// showAttention prints one head's weight matrix as a text heat map, rows
// being the position being read and columns the positions it drew from.
func (m *model) showAttention(prompt string) {
	ids := m.vocab.encode(prompt)

	x := autograd.Add(m.embedding.Forward(ids), autograd.PositionalEncoding(modelWidth, len(ids)))
	weights := m.blocks[0].Attention.Weights(0, m.blocks[0].AttnNorm.Forward(x), autograd.CausalMask(len(ids)))

	shades := []string{" ", ".", ":", "+", "*", "#"}

	fmt.Print("       ")
	for _, r := range prompt {
		fmt.Printf("%c", r)
	}
	fmt.Println("   <- attended to")

	for query := 0; query < len(ids); query++ {
		fmt.Printf("  %c -> ", []rune(prompt)[query])
		for key := 0; key < len(ids); key++ {
			w := weights.At(key, query)
			shade := int(w * float64(len(shades)))
			if shade >= len(shades) {
				shade = len(shades) - 1
			}
			fmt.Print(shades[shade])
		}
		fmt.Println()
	}
}
