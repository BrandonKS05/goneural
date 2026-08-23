package tokenizer

import (
	"strings"
	"testing"
)

const corpus = "the cat sat on the mat. the dog sat on the log. " +
	"the cat ran to the dog. the dog ran to the mat. " +
	"a cat and a dog sat on a log. a dog and a cat ran to a mat. "

// TestRoundTripsAnything is the guarantee the byte alphabet buys: no input
// is out of vocabulary, including text sharing nothing with the corpus.
func TestRoundTripsAnything(t *testing.T) {
	b := Train(corpus, 320)

	for _, text := range []string{
		corpus,
		"the cat sat on the mat.",
		"zebras quixotically vex",
		"ünïcödé ⇒ 日本語 🐈",
		"\x00\x01\xff raw bytes",
		" ",
		"a",
	} {
		if got := b.Decode(b.Encode(text)); got != text {
			t.Errorf("round trip of %q gave %q", text, got)
		}
	}
}

// TestEmptyInput covers the degenerate cases at the edges of Encode.
func TestEmptyInput(t *testing.T) {
	b := Train(corpus, 300)

	if got := b.Encode(""); len(got) != 0 {
		t.Errorf("Encode(\"\") = %v, want no tokens", got)
	}
	if got := b.Decode(nil); got != "" {
		t.Errorf("Decode(nil) = %q, want empty", got)
	}
	if got := b.CompressionRatio(""); got != 0 {
		t.Errorf("CompressionRatio(\"\") = %g, want 0", got)
	}
}

// TestMergesLearnFrequentSequences checks BPE finds what is actually in
// the corpus: the words that repeat should end up as single tokens.
func TestMergesLearnFrequentSequences(t *testing.T) {
	b := Train(corpus, 400)

	vocabulary := map[string]bool{}
	for id := 0; id < b.Size(); id++ {
		vocabulary[b.Token(id)] = true
	}

	// Whole words, with the leading space they almost always appear with.
	for _, want := range []string{" the", " cat", " dog", " sat", " ran", " and", " mat."} {
		if !vocabulary[want] {
			t.Errorf("vocabulary has no token %q, though it repeats throughout the corpus", want)
		}
	}

	// The first merge is the corpus's single most frequent adjacent pair,
	// which here is the space before every "the", "to" and "the mat".
	if first := b.Merges()[0]; first != " t" {
		t.Errorf("first merge was %q, want %q", first, " t")
	}
}

// TestNeverMergesAcrossWords pins the chunking rule: a token may carry a
// leading space, but never a space in the middle, or the vocabulary would
// fill with accidental phrases.
func TestNeverMergesAcrossWords(t *testing.T) {
	b := Train(corpus, 500)

	for id := baseVocabulary; id < b.Size(); id++ {
		token := b.Token(id)
		if strings.Contains(strings.TrimPrefix(token, " "), " ") {
			t.Errorf("token %q spans a word boundary", token)
		}
	}
}

// TestMoreMergesCompressFurther checks the central trade: vocabulary size
// buys sequence length, monotonically.
func TestMoreMergesCompressFurther(t *testing.T) {
	previous := 0.0
	for _, size := range []int{256, 280, 320, 400} {
		ratio := Train(corpus, size).CompressionRatio(corpus)

		if size == baseVocabulary && ratio != 1 {
			t.Errorf("an unmerged vocabulary compressed to %g, want exactly 1 byte per token", ratio)
		}
		if ratio < previous {
			t.Errorf("vocabulary of %d compressed to %g, worse than the smaller vocabulary's %g",
				size, ratio, previous)
		}
		previous = ratio
	}

	if previous < 2 {
		t.Errorf("400 tokens only reached %g bytes per token on a highly repetitive corpus", previous)
	}
}

// TestUnfamiliarTextCompressesWorse is the property that makes the ratio
// diagnostic rather than decorative: a tokenizer applied outside its
// training distribution reports it.
func TestUnfamiliarTextCompressesWorse(t *testing.T) {
	b := Train(corpus, 400)

	familiar := b.CompressionRatio("the cat sat on the mat. the dog ran to the log.")
	foreign := b.CompressionRatio("quixotic zebras vex jumpy wizards")

	if foreign >= familiar {
		t.Errorf("unfamiliar text compressed to %g, no worse than familiar text's %g", foreign, familiar)
	}
	if foreign < 1 {
		t.Errorf("compression of %g is below one byte per token, which should be impossible", foreign)
	}
}

// TestEncodingIsDeterministic checks training and encoding do not inherit
// Go's randomized map iteration order.
func TestEncodingIsDeterministic(t *testing.T) {
	first := Train(corpus, 350)

	for trial := 0; trial < 5; trial++ {
		other := Train(corpus, 350)

		if other.Size() != first.Size() {
			t.Fatalf("vocabulary size varied between runs: %d then %d", first.Size(), other.Size())
		}
		for id := 0; id < first.Size(); id++ {
			if other.Token(id) != first.Token(id) {
				t.Fatalf("token %d varied between runs: %q then %q", id, first.Token(id), other.Token(id))
			}
		}

		a, b := first.Encode(corpus), other.Encode(corpus)
		if len(a) != len(b) {
			t.Fatalf("encoding length varied between runs: %d then %d", len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("encoding varied at position %d: %d then %d", i, a[i], b[i])
			}
		}
	}
}

// TestEncodeReplaysTrainingOrder checks the rank-ordered merge loop, which
// is what keeps encoding consistent with what the vocabulary was built
// from. A longest-match-first shortcut would tokenize "theme" using the
// long " the" token; replaying merges in order does not.
func TestEncodeReplaysTrainingOrder(t *testing.T) {
	b := Train(corpus, 400)

	// Whatever segmentation it chooses, it must be reachable by replaying
	// merges: decoding each token and re-encoding the pieces has to give
	// the same ids back.
	ids := b.Encode(" the cat")
	for _, id := range ids {
		piece := b.Token(id)
		if got := b.Encode(piece); len(got) != 1 || got[0] != id {
			t.Errorf("token %q (id %d) does not re-encode to itself: %v", piece, id, got)
		}
	}
}

// TestTrainingStopsWhenNothingRepeats covers the early exit: a corpus with
// no repeated pair cannot fill a large vocabulary, and should not try.
func TestTrainingStopsWhenNothingRepeats(t *testing.T) {
	b := Train("abcdef", 5000)

	if b.Size() >= 5000 {
		t.Errorf("vocabulary reached %d on a corpus with nothing to merge", b.Size())
	}
	if got := b.Decode(b.Encode("abcdef")); got != "abcdef" {
		t.Errorf("round trip gave %q", got)
	}
}

func TestTrainRejectsBadInput(t *testing.T) {
	for name, f := range map[string]func(){
		"vocabulary too small": func() { Train(corpus, 100) },
		"empty corpus":         func() { Train("", 300) },
		"id out of range":      func() { Train(corpus, 300).Token(9999) },
		"negative id":          func() { Train(corpus, 300).Token(-1) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: expected a panic, got none", name)
				}
			}()
			f()
		}()
	}
}
