package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

// TestAttentionGradientsMatchNumeric differentiates a whole attention head
// against finite differences. Nothing in attention.go defines a backward
// pass -- the head is assembled from ordinary operations -- so this checks
// that composing them really does produce the derivative of the composite.
func TestAttentionGradientsMatchNumeric(t *testing.T) {
	rand.Seed(7)

	const (
		model     = 4
		head      = 3
		positions = 5
	)

	input := randomMatrix(model, positions)
	target := randomMatrix(model, positions)

	build := func(p []*Node) *Node {
		a := &AttentionHead{
			Query: p[0],
			Key:   p[1],
			Value: p[2],
			scale: 1 / math.Sqrt(head),
		}
		return MSELoss(a.Forward(Const(input), nil), Const(target))
	}

	checkGradients(t, "attention", build,
		randomMatrix(head, model),
		randomMatrix(head, model),
		randomMatrix(model, model))
}

// TestAttentionGradientsWithMask repeats the check with a causal mask in
// place, since the mask changes which paths gradient can flow along.
func TestAttentionGradientsWithMask(t *testing.T) {
	rand.Seed(7)

	const (
		model     = 3
		head      = 2
		positions = 4
	)

	input := randomMatrix(model, positions)
	target := randomMatrix(model, positions)
	mask := CausalMask(positions)

	build := func(p []*Node) *Node {
		a := &AttentionHead{
			Query: p[0],
			Key:   p[1],
			Value: p[2],
			scale: 1 / math.Sqrt(head),
		}
		return MSELoss(a.Forward(Const(input), mask), Const(target))
	}

	checkGradients(t, "masked attention", build,
		randomMatrix(head, model),
		randomMatrix(head, model),
		randomMatrix(model, model))
}

// TestAttentionWeightsAreDistributions checks the softmax axis is the one
// the operation intends: each query's weights over the keys must sum to 1.
func TestAttentionWeightsAreDistributions(t *testing.T) {
	rand.Seed(7)

	a := NewAttentionHead(4, 3)
	input := Const(randomMatrix(4, 5))
	weights := a.Weights(input, nil)

	for j := 0; j < weights.Columns; j++ {
		sum := 0.0
		for i := 0; i < weights.Rows; i++ {
			if v := weights.At(i, j); v < 0 || v > 1 {
				t.Fatalf("weight[%d][%d] = %g, outside [0, 1]", i, j, v)
			}
			sum += weights.At(i, j)
		}
		if math.Abs(sum-1) > 1e-12 {
			t.Errorf("query %d's weights sum to %g, want 1", j, sum)
		}
	}
}

// TestCausalMaskBlocksTheFuture is the property the mask exists for:
// changing a later position's input must not move an earlier position's
// output at all. Without the mask, it moves.
func TestCausalMaskBlocksTheFuture(t *testing.T) {
	rand.Seed(7)

	const positions = 4

	a := NewAttentionHead(3, 2)
	original := randomMatrix(3, positions)

	altered := original.Copy()
	for i := 0; i < altered.Rows; i++ {
		altered.Set(i, positions-1, altered.At(i, positions-1)+5) // rewrite the last position
	}

	mask := CausalMask(positions)
	before := a.Forward(Const(original), mask).Value
	after := a.Forward(Const(altered), mask).Value

	for j := 0; j < positions-1; j++ {
		for i := 0; i < before.Rows; i++ {
			if math.Abs(before.At(i, j)-after.At(i, j)) > 1e-12 {
				t.Errorf("masked output at position %d moved when the future changed", j)
			}
		}
	}

	// The same change without the mask must reach the earlier positions,
	// or the test above proves nothing.
	unmaskedBefore := a.Forward(Const(original), nil).Value
	unmaskedAfter := a.Forward(Const(altered), nil).Value

	moved := false
	for j := 0; j < positions-1 && !moved; j++ {
		for i := 0; i < unmaskedBefore.Rows; i++ {
			if math.Abs(unmaskedBefore.At(i, j)-unmaskedAfter.At(i, j)) > 1e-9 {
				moved = true
				break
			}
		}
	}
	if !moved {
		t.Error("without a mask the future still did not reach earlier positions")
	}
}

// lookupSample builds one instance of a retrieval task that attention can
// solve and a fixed-weight layer cannot: three content positions carry
// random symbols, exactly one of them is flagged, and a final query
// position must report the flagged symbol. Which position to read is
// decided by the data, not by the weights, so the model has to *look it
// up*.
//
// Layout of each position vector: symbols one-hot in rows 0..symbols-1,
// the flag in row symbols, the query marker in row symbols+1.
func lookupSample(symbols, contents int) (input matrix.Matrix, answer int) {
	positions := contents + 1
	model := symbols + 2

	input = matrix.New(model, positions, nil)
	flagged := rand.Intn(contents)

	for j := 0; j < contents; j++ {
		symbol := rand.Intn(symbols)
		input.Set(symbol, j, 1)
		if j == flagged {
			input.Set(symbols, j, 1)
			answer = symbol
		}
	}
	input.Set(symbols+1, positions-1, 1) // the query position

	return input, answer
}

// TestAttentionLearnsToRetrieve trains the head on that task end to end.
// Getting this right means the model learned to route information from a
// position it had to identify from content -- the thing attention is for.
func TestAttentionLearnsToRetrieve(t *testing.T) {
	rand.Seed(11)

	const (
		symbols   = 4
		contents  = 3
		model     = symbols + 2
		positions = contents + 1
		steps     = 600
	)

	a := NewAttentionHead(model, 8)

	// Reads the last column of the head's output: a fixed one-hot selector
	// is all "take position n" needs to be.
	selector := matrix.New(positions, 1, nil)
	selector.Set(positions-1, 0, 1)

	out := Param(matrix.New(symbols, model, nil).Map(func(_ float64, _, _ int) float64 {
		return rand.NormFloat64() * 0.3
	})).Named("readout")

	params := append(a.Parameters(), out)
	optimizer := Adam(0.05)

	predict := func(input matrix.Matrix) *Node {
		attended := a.Forward(Const(input), nil)
		return MatMul(out, MatMul(attended, Const(selector)))
	}

	for step := 0; step < steps; step++ {
		input, answer := lookupSample(symbols, contents)

		target := matrix.New(symbols, 1, nil)
		target.Set(answer, 0, 1)

		loss := SoftmaxCrossEntropy(predict(input), Const(target))

		ZeroGrad(params...)
		loss.Backward()
		optimizer.Step(params...)
	}

	correct := 0
	const trials = 200
	for i := 0; i < trials; i++ {
		input, answer := lookupSample(symbols, contents)

		logits := predict(input).Value
		best := 0
		for s := 1; s < symbols; s++ {
			if logits.At(s, 0) > logits.At(best, 0) {
				best = s
			}
		}
		if best == answer {
			correct++
		}
	}

	// Chance is 1 in symbols; anything near it means the head never learned
	// to find the flagged position.
	if accuracy := float64(correct) / trials; accuracy < 0.95 {
		t.Errorf("retrieval accuracy %.3f over %d trials, want the head to solve the task", accuracy, trials)
	}
}

// TestMaskedWeightsAreZeroAhead reads the mask's effect straight off the
// reported distribution: a query must place no weight on a later position.
func TestMaskedWeightsAreZeroAhead(t *testing.T) {
	rand.Seed(7)

	const positions = 5

	a := NewAttentionHead(3, 2)
	weights := a.Weights(Const(randomMatrix(3, positions)), CausalMask(positions))

	for query := 0; query < positions; query++ {
		sum := 0.0
		for key := 0; key < positions; key++ {
			w := weights.At(key, query)
			if key > query && w > 1e-9 {
				t.Errorf("query %d put weight %g on the later position %d", query, w, key)
			}
			sum += w
		}
		if math.Abs(sum-1) > 1e-12 {
			t.Errorf("query %d's masked weights sum to %g, want 1", query, sum)
		}
	}
}

func TestAttentionRejectsBadShapes(t *testing.T) {
	for name, f := range map[string]func(){
		"zero model":  func() { NewAttentionHead(0, 2) },
		"zero head":   func() { NewAttentionHead(4, 0) },
		"wrong input": func() { NewAttentionHead(4, 2).Forward(Const(matrix.New(3, 5, nil)), nil) },
		"wrong mask":  func() { NewAttentionHead(4, 2).Forward(Const(matrix.New(4, 5, nil)), CausalMask(3)) },
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
