// Command tokenizer trains a byte-pair encoding on a small corpus and
// shows what it learned: the merges it discovered, how much it compresses
// familiar and unfamiliar text, and how a sentence gets segmented.
//
// It is the counterpart to examples/tinylm, which works one character at a
// time. A character model spends its context window on spelling; a subword
// model spends it on meaning. The compression ratio printed below is
// exactly how much further the same window reaches.
package main

import (
	"fmt"
	"strings"

	"github.com/BrandonKS05/goneural/tokenizer"
)

// corpus is a few paragraphs about the thing this repository does, chosen
// because a tokenizer's vocabulary should look like whatever it will be
// asked to read.
const corpus = `a neural network learns by adjusting weights. the network
makes a prediction, the loss measures how wrong the prediction was, and the
gradient of that loss tells every weight which way to move. training repeats
this until the loss stops falling.

the gradient is computed by backpropagation, which applies the chain rule
backwards through the network. each layer receives the gradient of the loss
with respect to its output, and passes back the gradient with respect to its
input. an optimizer turns those gradients into weight updates.

a convolutional network shares weights across positions, so a feature learned
in one place is detected everywhere. a recurrent network carries state across
a sequence. an attention layer computes where to look, so information moves
between positions the network chooses rather than positions fixed in advance.`

const vocabularySize = 512

func main() {
	bpe := tokenizer.Train(corpus, vocabularySize)

	fmt.Printf("trained on %d bytes, vocabulary of %d (%d merges on top of the 256 byte values)\n\n",
		len(corpus), bpe.Size(), bpe.Size()-256)

	merges := bpe.Merges()
	fmt.Println("the first merges it found, in order:")
	for i := 0; i < 12 && i < len(merges); i++ {
		fmt.Printf("  %2d  %q\n", i+1, merges[i])
	}

	fmt.Println("\nand some of the longest tokens it ended up with:")
	for _, token := range longest(bpe, 8) {
		fmt.Printf("  %q\n", token)
	}

	fmt.Println("\ncompression, in bytes per token:")
	for _, sample := range []struct {
		label string
		text  string
	}{
		{"the training corpus", corpus},
		{"a familiar sentence", "the gradient of the loss tells every weight which way to move."},
		{"related but unseen", "the optimizer computes a weight update from the gradient."},
		{"a different subject", "quixotic zebras vex jumpy wizards in strange weather."},
	} {
		fmt.Printf("  %-22s %.2f\n", sample.label, bpe.CompressionRatio(sample.text))
	}

	fmt.Println("\nhow it segments a sentence (| marks a token boundary):")
	for _, sentence := range []string{
		"the network learns by adjusting weights.",
		"an attention layer computes where to look.",
		"quixotic zebras vex jumpy wizards.",
	} {
		fmt.Printf("\n  %s\n  %s\n", sentence, segment(bpe, sentence))
	}

	// What the compression is actually for.
	window := 64
	fmt.Printf("\na %d-token context window covers about %.0f characters of this corpus,\n",
		window, float64(window)*bpe.CompressionRatio(corpus))
	fmt.Printf("against %d for a character-level model like examples/tinylm.\n", window)
}

// segment renders a tokenization with visible boundaries.
func segment(bpe *tokenizer.BPE, text string) string {
	var parts []string
	for _, id := range bpe.Encode(text) {
		parts = append(parts, bpe.Token(id))
	}
	return "|" + strings.Join(parts, "|") + "|"
}

// longest returns the n longest tokens in the vocabulary, which is a quick
// read on what the corpus repeats most.
func longest(bpe *tokenizer.BPE, n int) []string {
	var tokens []string
	for id := 256; id < bpe.Size(); id++ {
		tokens = append(tokens, bpe.Token(id))
	}

	// A simple selection: repeatedly take the longest remaining.
	var out []string
	for len(out) < n && len(tokens) > 0 {
		best := 0
		for i, token := range tokens {
			if len(token) > len(tokens[best]) {
				best = i
			}
		}
		out = append(out, tokens[best])
		tokens = append(tokens[:best], tokens[best+1:]...)
	}

	return out
}
