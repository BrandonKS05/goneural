package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

// TestLayerNormGradients is the check LayerNorm's hand-derived backward
// pass exists to be held to: the two correction terms are easy to get
// subtly wrong and impossible to notice without this.
func TestLayerNormGradients(t *testing.T) {
	rand.Seed(7)

	const (
		features  = 5
		positions = 3
	)

	target := randomMatrix(features, positions)

	build := func(p []*Node) *Node {
		norm := &LayerNorm{Gain: p[1], Shift: p[2], Epsilon: 1e-5}
		return MSELoss(norm.Forward(p[0]), Const(target))
	}

	checkGradients(t, "layernorm", build,
		randomMatrix(features, positions),
		randomMatrix(features, 1),
		randomMatrix(features, 1))
}

// TestLayerNormNormalizes checks the forward pass does what it claims:
// each column, not each row, comes out with zero mean and unit variance.
func TestLayerNormNormalizes(t *testing.T) {
	rand.Seed(7)

	const (
		features  = 6
		positions = 4
	)

	// Columns deliberately given wildly different scales and offsets.
	input := matrix.New(features, positions, nil).Map(func(_ float64, _, j int) float64 {
		return rand.NormFloat64()*math.Pow(10, float64(j)) + float64(j*100)
	})

	norm := NewLayerNorm(features)
	out := norm.Forward(Const(input)).Value

	for j := 0; j < positions; j++ {
		mean := 0.0
		for i := 0; i < features; i++ {
			mean += out.At(i, j)
		}
		mean /= features

		variance := 0.0
		for i := 0; i < features; i++ {
			variance += (out.At(i, j) - mean) * (out.At(i, j) - mean)
		}
		variance /= features

		if math.Abs(mean) > 1e-9 {
			t.Errorf("position %d has mean %g, want 0", j, mean)
		}
		if math.Abs(variance-1) > 1e-4 {
			t.Errorf("position %d has variance %g, want 1", j, variance)
		}
	}
}

// TestLayerNormIsPerPosition pins the property that separates it from
// batch normalization: one position's output must not depend on any other
// position's values.
func TestLayerNormIsPerPosition(t *testing.T) {
	rand.Seed(7)

	input := randomMatrix(4, 3)
	altered := input.Copy()
	for i := 0; i < altered.Rows; i++ {
		altered.Set(i, 2, altered.At(i, 2)*50)
	}

	norm := NewLayerNorm(4)
	before := norm.Forward(Const(input)).Value
	after := norm.Forward(Const(altered)).Value

	for j := 0; j < 2; j++ {
		for i := 0; i < 4; i++ {
			if math.Abs(before.At(i, j)-after.At(i, j)) > 1e-12 {
				t.Errorf("position %d moved when position 2 changed", j)
			}
		}
	}
}

// TestLayerNormStartsAsTheIdentityTransform checks a fresh normalizer only
// normalizes -- unit gain, zero shift -- so inserting one into a working
// model changes nothing beyond the normalization itself.
func TestLayerNormStartsAsTheIdentityTransform(t *testing.T) {
	norm := NewLayerNorm(3)

	for i := 0; i < 3; i++ {
		if got := norm.Gain.Value.At(i, 0); got != 1 {
			t.Errorf("gain[%d] = %g, want 1", i, got)
		}
		if got := norm.Shift.Value.At(i, 0); got != 0 {
			t.Errorf("shift[%d] = %g, want 0", i, got)
		}
	}
}

// TestLayerNormHandlesConstantPosition covers the zero-variance guard.
func TestLayerNormHandlesConstantPosition(t *testing.T) {
	input := matrix.New(4, 1, nil).Map(func(_ float64, _, _ int) float64 { return 7 })

	for _, v := range NewLayerNorm(4).Forward(Const(input)).Value.Flatten() {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("constant position produced %g", v)
		}
	}
}

func TestLinearGradients(t *testing.T) {
	rand.Seed(7)

	target := randomMatrix(2, 4)

	build := func(p []*Node) *Node {
		layer := &Linear{Weight: p[0], Bias: p[1]}
		return MSELoss(layer.Forward(p[2]), Const(target))
	}

	checkGradients(t, "linear", build,
		randomMatrix(2, 3),
		randomMatrix(2, 1),
		randomMatrix(3, 4))
}

// TestDropoutPreservesExpectedValue checks the inverted scaling: the mean
// activation must survive the masking, which is what lets inference skip
// dropout entirely.
func TestDropoutPreservesExpectedValue(t *testing.T) {
	rand.Seed(7)

	ones := matrix.New(40, 40, nil).Map(func(_ float64, _, _ int) float64 { return 1 })

	total, trials := 0.0, 40
	for i := 0; i < trials; i++ {
		total += Dropout(Const(ones), 0.5, true).Value.Mean()
	}

	if mean := total / float64(trials); math.Abs(mean-1) > 0.02 {
		t.Errorf("mean activation under dropout = %g, want about 1", mean)
	}
}

// TestDropoutIsInertAtInference pins the training flag.
func TestDropoutIsInertAtInference(t *testing.T) {
	rand.Seed(7)

	input := randomMatrix(6, 6)
	if got := Dropout(Const(input), 0.5, false); !got.Value.Equal(input) {
		t.Error("dropout altered the values with training false")
	}
	if got := Dropout(Const(input), 0, true); !got.Value.Equal(input) {
		t.Error("a rate of zero altered the values")
	}
}

// TestDropoutGradientFollowsTheMask checks gradient reaches exactly the
// units that survived, with the same scaling the forward pass used.
func TestDropoutGradientFollowsTheMask(t *testing.T) {
	rand.Seed(7)

	x := Param(randomMatrix(8, 4))
	dropped := Dropout(x, 0.5, true)

	Sum(dropped).Backward()

	for i := 0; i < x.Value.Rows; i++ {
		for j := 0; j < x.Value.Columns; j++ {
			zeroed := dropped.Value.At(i, j) == 0
			if zeroed && x.Grad.At(i, j) != 0 {
				t.Errorf("dropped unit (%d, %d) still received gradient %g", i, j, x.Grad.At(i, j))
			}
			if !zeroed && math.Abs(x.Grad.At(i, j)-2) > 1e-12 {
				t.Errorf("surviving unit (%d, %d) received %g, want the 1/keep scale of 2",
					i, j, x.Grad.At(i, j))
			}
		}
	}
}

func TestScaleRowsAndConcatGradients(t *testing.T) {
	rand.Seed(7)

	checkGradients(t, "scalerows", func(p []*Node) *Node {
		return Sum(Mul(ScaleRows(p[0], p[1]), p[2]))
	}, randomMatrix(3, 4), randomMatrix(3, 1), randomMatrix(3, 4))

	checkGradients(t, "concatrows", func(p []*Node) *Node {
		return Sum(Mul(ConcatRows(p[0], p[1], p[2]), p[3]))
	}, randomMatrix(2, 3), randomMatrix(1, 3), randomMatrix(3, 3), randomMatrix(6, 3))
}

// TestConcatRowsStacksInOrder checks the forward layout, since a wrong
// order would still pass a gradient check.
func TestConcatRowsStacksInOrder(t *testing.T) {
	top := Const(matrix.New(1, 2, [][]float64{{1, 2}}))
	bottom := Const(matrix.New(2, 2, [][]float64{{3, 4}, {5, 6}}))

	got := ConcatRows(top, bottom).Value
	want := matrix.New(3, 2, [][]float64{{1, 2}, {3, 4}, {5, 6}})

	if !got.Equal(want) {
		t.Errorf("ConcatRows gave\n%v\nwant\n%v", got, want)
	}
}

func TestLayersRejectBadShapes(t *testing.T) {
	for name, f := range map[string]func(){
		"zero linear":     func() { NewLinear(0, 3) },
		"zero layernorm":  func() { NewLayerNorm(0) },
		"wrong features":  func() { NewLayerNorm(3).Forward(Const(matrix.New(4, 2, nil))) },
		"bad rate":        func() { Dropout(Const(matrix.New(2, 2, nil)), 1, true) },
		"bad scale shape": func() { ScaleRows(Const(matrix.New(3, 2, nil)), Const(matrix.New(2, 1, nil))) },
		"ragged concat":   func() { ConcatRows(Const(matrix.New(1, 2, nil)), Const(matrix.New(1, 3, nil))) },
		"empty concat":    func() { ConcatRows() },
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
