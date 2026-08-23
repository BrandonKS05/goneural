// Package tokenizer implements byte-pair encoding, the step that turns
// text into the integer ids a model actually consumes.
//
// The choice sits between two bad extremes. A word vocabulary is compact
// but brittle: every unseen word, typo, or inflection collapses to a
// single "unknown" token, and the vocabulary itself grows without bound.
// Characters never fail that way, but they carry so little meaning each
// that a model spends its capacity relearning spelling and its context
// window on a handful of words.
//
// Byte-pair encoding (Sennrich et al., 2016, adapting a 1994 compression
// scheme) interpolates between them, and does it from the data rather than
// from a rulebook. Start with the raw bytes; repeatedly find the most
// frequent adjacent pair in the corpus and fuse it into a new symbol.
// Common words end up as one token, rare ones decompose into familiar
// fragments, and nothing is ever out of vocabulary -- the base alphabet is
// all 256 byte values, so any input at all can be represented.
package tokenizer

import (
	"fmt"
	"sort"
	"strings"
)

// baseVocabulary is the 256 single-byte tokens every BPE starts from.
// Working in bytes rather than runes is what makes the "no unknown token"
// guarantee unconditional: an emoji, a control character, or invalid UTF-8
// is still a sequence of bytes.
const baseVocabulary = 256

// pair is two adjacent token ids, the unit BPE merges.
type pair struct {
	left  int
	right int
}

// BPE is a trained byte-pair encoding: a vocabulary of tokens and the
// ordered list of merges that produced it. It is read-only once trained
// and safe for concurrent use.
type BPE struct {
	// tokens maps an id to the literal bytes it stands for.
	tokens []string

	// ranks records the order merges were learned in. Order is everything
	// at encoding time: applying a late merge before an early one would
	// produce a different segmentation than training ever saw.
	ranks map[pair]int

	// merged maps a pair to the id it fuses into.
	merged map[pair]int
}

// Train learns a vocabulary of the requested size from a corpus. size must
// be at least 256, since the byte alphabet is not optional; training stops
// early if no adjacent pair repeats often enough to be worth a token, so
// the result may be smaller than asked for.
func Train(corpus string, size int) *BPE {
	if size < baseVocabulary {
		panic(fmt.Sprintf("tokenizer: vocabulary of %d is below the %d byte values", size, baseVocabulary))
	}
	if corpus == "" {
		panic("tokenizer: cannot train on an empty corpus")
	}

	b := &BPE{
		tokens: make([]string, baseVocabulary),
		ranks:  map[pair]int{},
		merged: map[pair]int{},
	}
	for i := range b.tokens {
		b.tokens[i] = string([]byte{byte(i)})
	}

	// Counting over unique chunks with their frequencies, rather than over
	// the raw corpus, is what keeps training tractable: a word appearing a
	// thousand times is scanned once and weighted.
	counts := map[string]int{}
	for _, chunk := range split(corpus) {
		counts[chunk]++
	}

	type entry struct {
		symbols []int
		count   int
	}

	// Sorted, so training is deterministic despite the map.
	keys := make([]string, 0, len(counts))
	for chunk := range counts {
		keys = append(keys, chunk)
	}
	sort.Strings(keys)

	words := make([]entry, len(keys))
	for i, chunk := range keys {
		symbols := make([]int, 0, len(chunk))
		for _, byteValue := range []byte(chunk) {
			symbols = append(symbols, int(byteValue))
		}
		words[i] = entry{symbols: symbols, count: counts[chunk]}
	}

	for len(b.tokens) < size {
		// Count every adjacent pair across the corpus.
		frequency := map[pair]int{}
		for _, w := range words {
			for i := 0; i+1 < len(w.symbols); i++ {
				frequency[pair{w.symbols[i], w.symbols[i+1]}] += w.count
			}
		}

		best, bestCount := pair{}, 0
		for p, count := range frequency {
			// Ties broken by id keeps the vocabulary reproducible; map
			// iteration order alone would not be.
			if count > bestCount ||
				(count == bestCount && (p.left < best.left || (p.left == best.left && p.right < best.right))) {
				best, bestCount = p, count
			}
		}

		// A pair seen once buys nothing: it would spend a vocabulary slot
		// to shorten the corpus by a single token.
		if bestCount < 2 {
			break
		}

		id := len(b.tokens)
		b.tokens = append(b.tokens, b.tokens[best.left]+b.tokens[best.right])
		b.ranks[best] = len(b.ranks)
		b.merged[best] = id

		for i := range words {
			words[i].symbols = apply(words[i].symbols, best, id)
		}
	}

	return b
}

// apply replaces every occurrence of a pair in a symbol sequence with the
// merged id, scanning left to right so overlapping occurrences (as in
// "aaa") consume greedily rather than double-counting.
func apply(symbols []int, p pair, id int) []int {
	out := make([]int, 0, len(symbols))
	for i := 0; i < len(symbols); {
		if i+1 < len(symbols) && symbols[i] == p.left && symbols[i+1] == p.right {
			out = append(out, id)
			i += 2
			continue
		}
		out = append(out, symbols[i])
		i++
	}
	return out
}

// split breaks text into the chunks merges are allowed to run inside. A
// space starts a new chunk and stays attached to the word that follows it,
// so " the" can become a single token while "the cat" can never fuse
// across the gap -- which stops the vocabulary from filling up with
// accidental multi-word phrases.
func split(text string) []string {
	var chunks []string

	start := 0
	for i := 1; i < len(text); i++ {
		if text[i] == ' ' {
			chunks = append(chunks, text[start:i])
			start = i
		}
	}
	if start < len(text) {
		chunks = append(chunks, text[start:])
	}

	return chunks
}

// Size returns the number of tokens in the vocabulary.
func (b *BPE) Size() int {
	return len(b.tokens)
}

// Token returns the literal text an id stands for.
func (b *BPE) Token(id int) string {
	if id < 0 || id >= len(b.tokens) {
		panic("tokenizer: token id out of range")
	}
	return b.tokens[id]
}

// Merges returns the learned merges in the order they were found, each as
// the token text it produced. Reading the first few is the quickest way to
// see what a corpus is made of.
func (b *BPE) Merges() []string {
	out := make([]string, len(b.ranks))
	for p, rank := range b.ranks {
		out[rank] = b.tokens[b.merged[p]]
	}
	return out
}

// Encode turns text into token ids.
//
// The merges are replayed in the order they were learned: at each step the
// lowest-ranked pair still present is fused, everywhere it appears, and
// the scan restarts. Greedily taking the longest match instead would be
// faster and wrong -- it can produce segmentations the training data never
// contained, splitting a word into pieces the model has never seen
// together.
func (b *BPE) Encode(text string) []int {
	var out []int

	for _, chunk := range split(text) {
		symbols := make([]int, 0, len(chunk))
		for _, byteValue := range []byte(chunk) {
			symbols = append(symbols, int(byteValue))
		}

		for len(symbols) > 1 {
			bestRank, best := -1, pair{}
			for i := 0; i+1 < len(symbols); i++ {
				p := pair{symbols[i], symbols[i+1]}
				if rank, ok := b.ranks[p]; ok && (bestRank == -1 || rank < bestRank) {
					bestRank, best = rank, p
				}
			}
			if bestRank == -1 {
				break
			}

			symbols = apply(symbols, best, b.merged[best])
		}

		out = append(out, symbols...)
	}

	return out
}

// Decode turns token ids back into text. It is an exact inverse of Encode
// for any input at all, since every token is a literal byte string and the
// base alphabet covers all 256 of them.
func (b *BPE) Decode(ids []int) string {
	var out strings.Builder
	for _, id := range ids {
		out.WriteString(b.Token(id))
	}
	return out.String()
}

// CompressionRatio reports bytes per token when this encoding is applied
// to the given text -- the practical measure of what the vocabulary bought.
// It is 1 for an untrained (byte-level) encoding and climbs as merges
// absorb common sequences; text unlike the training corpus scores lower,
// which is exactly the tokenizer telling you it is out of its depth.
func (b *BPE) CompressionRatio(text string) float64 {
	encoded := b.Encode(text)
	if len(encoded) == 0 {
		return 0
	}
	return float64(len(text)) / float64(len(encoded))
}
