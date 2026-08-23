package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

// TestTransformerBlockGradients differentiates a whole block -- two
// normalizations, four heads, a residual stream and a widening
// feed-forward pair -- against finite differences. Every parameter in it
// has to agree, which is a strong check that the pieces compose.
func TestTransformerBlockGradients(t *testing.T) {
	rand.Seed(7)

	const (
		model     = 4
		heads     = 2
		positions = 3
	)

	block := NewTransformerBlock(model, heads)
	params := block.Parameters()

	input := randomMatrix(model, positions)
	target := randomMatrix(model, positions)
	mask := CausalMask(positions)

	values := make([]matrix.Matrix, len(params))
	for i, p := range params {
		values[i] = p.Value.Copy()
	}

	build := func(p []*Node) *Node {
		// Rebuild the block around the supplied parameter nodes, in the
		// same order Parameters returned them.
		rebuilt := &TransformerBlock{
			Attention: &MultiHeadAttention{
				Query: p[0:heads],
				Key:   p[heads : 2*heads],
				Value: p[2*heads : 3*heads],
				Out:   &Linear{Weight: p[3*heads], Bias: p[3*heads+1]},
				scale: 1 / math.Sqrt(model/heads),
			},
			AttnNorm: &LayerNorm{Gain: p[3*heads+2], Shift: p[3*heads+3], Epsilon: 1e-5},
			FFNorm:   &LayerNorm{Gain: p[3*heads+4], Shift: p[3*heads+5], Epsilon: 1e-5},
			Up:       &Linear{Weight: p[3*heads+6], Bias: p[3*heads+7]},
			Down:     &Linear{Weight: p[3*heads+8], Bias: p[3*heads+9]},
		}

		return MSELoss(rebuilt.Forward(Const(input), mask, false), Const(target))
	}

	checkGradients(t, "transformer block", build, values...)
}

func TestEmbeddingGradients(t *testing.T) {
	rand.Seed(7)

	// Token 1 appears twice, so its column must collect gradient from both
	// positions.
	ids := []int{1, 3, 1, 0}
	target := randomMatrix(3, len(ids))

	build := func(p []*Node) *Node {
		table := &Embedding{Table: p[0]}
		return MSELoss(table.Forward(ids), Const(target))
	}

	checkGradients(t, "embedding", build, randomMatrix(3, 5))
}

// TestEmbeddingLooksUpColumns checks the forward layout, and that unused
// entries receive no gradient at all.
func TestEmbeddingLooksUpColumns(t *testing.T) {
	rand.Seed(7)

	table := NewEmbedding(4, 3)
	out := table.Forward([]int{2, 0})

	for i := 0; i < 3; i++ {
		if got, want := out.Value.At(i, 0), table.Table.Value.At(i, 2); got != want {
			t.Errorf("position 0 row %d = %g, want column 2's %g", i, got, want)
		}
		if got, want := out.Value.At(i, 1), table.Table.Value.At(i, 0); got != want {
			t.Errorf("position 1 row %d = %g, want column 0's %g", i, got, want)
		}
	}

	Sum(out).Backward()

	for _, unused := range []int{1, 3} {
		for i := 0; i < 3; i++ {
			if got := table.Table.Grad.At(i, unused); got != 0 {
				t.Errorf("unused entry %d row %d got gradient %g, want 0", unused, i, got)
			}
		}
	}
}

// TestMultiHeadOutputShapeAndWeights checks the concatenation comes back
// at model width and that each head reports its own distribution.
func TestMultiHeadOutputShapeAndWeights(t *testing.T) {
	rand.Seed(7)

	const (
		model     = 8
		heads     = 4
		positions = 5
	)

	a := NewMultiHeadAttention(model, heads)
	input := Const(randomMatrix(model, positions))

	out := a.Forward(input, nil).Value
	if out.Rows != model || out.Columns != positions {
		t.Fatalf("output is %dx%d, want %dx%d", out.Rows, out.Columns, model, positions)
	}
	if a.Heads() != heads {
		t.Errorf("Heads() = %d, want %d", a.Heads(), heads)
	}

	for h := 0; h < heads; h++ {
		w := a.Weights(h, input, nil)
		if w.Rows != positions || w.Columns != positions {
			t.Fatalf("head %d weights are %dx%d, want %dx%d", h, w.Rows, w.Columns, positions, positions)
		}
		for j := 0; j < positions; j++ {
			sum := 0.0
			for i := 0; i < positions; i++ {
				sum += w.At(i, j)
			}
			if math.Abs(sum-1) > 1e-12 {
				t.Errorf("head %d query %d sums to %g, want 1", h, j, sum)
			}
		}
	}
}

// TestHeadsAreIndependent checks the heads really are separate: they start
// from different random projections, so on the same input they must
// produce different attention patterns.
func TestHeadsAreIndependent(t *testing.T) {
	rand.Seed(7)

	a := NewMultiHeadAttention(8, 4)
	input := Const(randomMatrix(8, 5))

	first := a.Weights(0, input, nil)
	for h := 1; h < a.Heads(); h++ {
		if a.Weights(h, input, nil).ApproxEqual(first, 1e-6) {
			t.Errorf("head %d produced the same pattern as head 0", h)
		}
	}
}

// TestResidualStreamPassesThrough pins the pre-norm arrangement: with the
// sub-layer outputs dropped entirely, the block is the identity, which is
// the property that keeps a deep stack's gradient path clear.
func TestResidualStreamPassesThrough(t *testing.T) {
	rand.Seed(7)

	block := NewTransformerBlock(4, 2)

	// Zero the two projections that write back into the residual stream,
	// which is exactly what a block contributes nothing looks like.
	block.Attention.Out.Weight.Value = block.Attention.Out.Weight.Value.Scale(0)
	block.Down.Weight.Value = block.Down.Weight.Value.Scale(0)

	input := randomMatrix(4, 3)
	out := block.Forward(Const(input), nil, false).Value

	if !out.ApproxEqual(input, 1e-12) {
		t.Errorf("a contributionless block changed its input:\ngot %v\nwant %v", out, input)
	}
}

// TestPositionalEncodingIsBoundedAndDistinct checks the encoding stays in
// range and actually distinguishes positions -- an encoding that repeated
// would be worse than none.
func TestPositionalEncodingIsBoundedAndDistinct(t *testing.T) {
	const (
		width     = 16
		positions = 32
	)

	encoding := PositionalEncoding(width, positions).Value

	for _, v := range encoding.Flatten() {
		if v < -1 || v > 1 {
			t.Fatalf("encoding value %g outside [-1, 1]", v)
		}
	}

	for a := 0; a < positions; a++ {
		for b := a + 1; b < positions; b++ {
			same := true
			for i := 0; i < width && same; i++ {
				if math.Abs(encoding.At(i, a)-encoding.At(i, b)) > 1e-9 {
					same = false
				}
			}
			if same {
				t.Fatalf("positions %d and %d have identical encodings", a, b)
			}
		}
	}
}

// TestPositionalEncodingIsShiftLinear checks the property the sinusoids
// are chosen for: the encoding at p+k is a fixed linear map of the one at
// p, the same map for every p. Verified here on one frequency pair, where
// that map is a plain rotation.
func TestPositionalEncodingIsShiftLinear(t *testing.T) {
	encoding := PositionalEncoding(4, 10).Value

	const k = 3
	frequency := 1.0 // the first row pair has frequency 1
	cos, sin := math.Cos(k*frequency), math.Sin(k*frequency)

	for p := 0; p+k < 10; p++ {
		wantSin := encoding.At(0, p)*cos + encoding.At(1, p)*sin
		wantCos := encoding.At(1, p)*cos - encoding.At(0, p)*sin

		if math.Abs(encoding.At(0, p+k)-wantSin) > 1e-9 || math.Abs(encoding.At(1, p+k)-wantCos) > 1e-9 {
			t.Errorf("shifting position %d by %d is not the expected rotation", p, k)
		}
	}
}

func TestTransformerRejectsBadShapes(t *testing.T) {
	for name, f := range map[string]func(){
		"indivisible model": func() { NewMultiHeadAttention(6, 4) },
		"zero heads":        func() { NewMultiHeadAttention(4, 0) },
		"head out of range": func() { NewMultiHeadAttention(4, 2).Weights(2, Const(matrix.New(4, 3, nil)), nil) },
		"wrong input":       func() { NewMultiHeadAttention(4, 2).Forward(Const(matrix.New(3, 3, nil)), nil) },
		"wrong mask":        func() { NewMultiHeadAttention(4, 2).Forward(Const(matrix.New(4, 3, nil)), CausalMask(2)) },
		"zero vocab":        func() { NewEmbedding(0, 4) },
		"empty lookup":      func() { NewEmbedding(4, 4).Forward(nil) },
		"id out of range":   func() { NewEmbedding(4, 4).Forward([]int{9}) },
		"zero encoding":     func() { PositionalEncoding(0, 4) },
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
