package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

// TestConv2DGradients differentiates a convolution against finite
// differences, including the input's gradient -- the scatter that has to
// undo im2col's overlapping gather.
func TestConv2DGradients(t *testing.T) {
	rand.Seed(7)

	for _, tc := range []struct {
		name                           string
		in                             ImageShape
		filters, size, stride, padding int
	}{
		{"single channel", ImageShape{1, 5, 5}, 2, 3, 1, 0},
		{"padded", ImageShape{1, 4, 4}, 3, 3, 1, 1},
		{"strided", ImageShape{2, 6, 6}, 2, 3, 2, 0},
		{"multi channel", ImageShape{3, 4, 5}, 2, 2, 1, 0},
	} {
		conv := NewConv2D(tc.in, tc.filters, tc.size, tc.stride, tc.padding)
		out := conv.Out()
		target := randomMatrix(out.Rows(), out.Width)

		build := func(p []*Node) *Node {
			layer := &Conv2D{
				Kernels: p[0], Bias: p[1],
				In: tc.in, Size: tc.size, Stride: tc.stride, Padding: tc.padding,
			}
			return MSELoss(layer.Forward(p[2]), Const(target))
		}

		checkGradients(t, "conv2d "+tc.name, build,
			conv.Kernels.Value.Copy(),
			randomMatrix(tc.filters, 1),
			randomMatrix(tc.in.Rows(), tc.in.Width))
	}
}

// TestConv2DMatchesDirectConvolution checks the im2col shortcut against
// the definition, computed here by an explicit loop nest.
func TestConv2DMatchesDirectConvolution(t *testing.T) {
	rand.Seed(7)

	in := ImageShape{Channels: 2, Height: 5, Width: 4}
	const (
		filters = 3
		size    = 3
	)

	conv := NewConv2D(in, filters, size, 1, 0)
	image := randomMatrix(in.Rows(), in.Width)
	got := conv.Forward(Const(image)).Value

	out := conv.Out()
	for f := 0; f < filters; f++ {
		for oy := 0; oy < out.Height; oy++ {
			for ox := 0; ox < out.Width; ox++ {
				want := conv.Bias.Value.At(f, 0)

				tap := 0
				for ch := 0; ch < in.Channels; ch++ {
					for ky := 0; ky < size; ky++ {
						for kx := 0; kx < size; kx++ {
							want += conv.Kernels.Value.At(f, tap) *
								image.At(ch*in.Height+oy+ky, ox+kx)
							tap++
						}
					}
				}

				if g := got.At(f*out.Height+oy, ox); math.Abs(g-want) > 1e-12 {
					t.Fatalf("filter %d at (%d, %d) = %g, want %g", f, oy, ox, g, want)
				}
			}
		}
	}
}

// TestConv2DDetectsAnEdgeAnywhere is the property convolution exists for:
// one kernel finds the same feature wherever it appears, with no extra
// parameters per location.
func TestConv2DDetectsAnEdgeAnywhere(t *testing.T) {
	in := ImageShape{Channels: 1, Height: 6, Width: 6}
	conv := NewConv2D(in, 1, 3, 1, 0)

	// A vertical-edge detector: negative column, zero, positive column.
	kernel := []float64{-1, 0, 1, -1, 0, 1, -1, 0, 1}
	for i, v := range kernel {
		conv.Kernels.Value.Set(0, i, v)
	}
	conv.Bias.Value.Set(0, 0, 0)

	for edge := 1; edge < 5; edge++ {
		// An image that is dark left of the edge and bright right of it.
		image := matrix.New(in.Rows(), in.Width, nil).
			Map(func(_ float64, _, x int) float64 {
				if x >= edge {
					return 1
				}
				return 0
			})

		response := conv.Forward(Const(image)).Value

		best, at := math.Inf(-1), 0
		for x := 0; x < response.Columns; x++ {
			if v := response.At(0, x); v > best {
				best, at = v, x
			}
		}

		// The response peaks wherever the kernel straddles the transition:
		// its three columns run at .. at+2, so the edge has to fall inside
		// them. Two placements can tie, and either is a correct detection.
		if at >= edge || at+2 < edge {
			t.Errorf("edge at column %d detected at %d, which does not straddle it", edge, at)
		}
		if math.Abs(best-3) > 1e-12 {
			t.Errorf("edge response %g, want the full 3", best)
		}
	}
}

// TestConv2DPaddingPreservesSize pins the shape arithmetic.
func TestConv2DPaddingPreservesSize(t *testing.T) {
	in := ImageShape{Channels: 1, Height: 7, Width: 9}

	same := NewConv2D(in, 4, 3, 1, 1).Out()
	if same.Height != in.Height || same.Width != in.Width {
		t.Errorf("padded 3x3 gave %dx%d, want the input's %dx%d",
			same.Height, same.Width, in.Height, in.Width)
	}
	if same.Channels != 4 {
		t.Errorf("output has %d channels, want one per filter", same.Channels)
	}

	strided := NewConv2D(in, 2, 3, 2, 0).Out()
	if strided.Height != 3 || strided.Width != 4 {
		t.Errorf("strided conv gave %dx%d, want 3x4", strided.Height, strided.Width)
	}
}

func TestMaxPoolGradients(t *testing.T) {
	rand.Seed(7)

	in := ImageShape{Channels: 2, Height: 4, Width: 4}
	target := randomMatrix(2*2, 2)

	checkGradients(t, "maxpool", func(p []*Node) *Node {
		return MSELoss(MaxPool2D(p[0], in, 2, 2), Const(target))
	}, randomMatrix(in.Rows(), in.Width))
}

// TestMaxPoolTakesTheWindowMaximum checks the forward values and that the
// gradient reaches only the winners.
func TestMaxPoolTakesTheWindowMaximum(t *testing.T) {
	in := ImageShape{Channels: 1, Height: 4, Width: 4}

	image := matrix.New(4, 4, [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 1, 2, 3},
		{4, 5, 6, 7},
	})

	x := Param(image)
	pooled := MaxPool2D(x, in, 2, 2)

	want := matrix.New(2, 2, [][]float64{{6, 8}, {9, 7}})
	if !pooled.Value.Equal(want) {
		t.Fatalf("pooled to\n%v\nwant\n%v", pooled.Value, want)
	}

	Sum(pooled).Backward()

	winners := map[[2]int]bool{{1, 1}: true, {1, 3}: true, {2, 0}: true, {3, 3}: true}
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			got := x.Grad.At(i, j)
			if winners[[2]int{i, j}] && got != 1 {
				t.Errorf("winner (%d, %d) got gradient %g, want 1", i, j, got)
			}
			if !winners[[2]int{i, j}] && got != 0 {
				t.Errorf("non-winner (%d, %d) got gradient %g, want 0", i, j, got)
			}
		}
	}
}

// TestMaxPoolIsShiftInvariant is the invariance claim, checked directly: a
// one-pixel shift inside a pooling window must not change the output.
func TestMaxPoolIsShiftInvariant(t *testing.T) {
	in := ImageShape{Channels: 1, Height: 2, Width: 4}

	left := matrix.New(2, 4, [][]float64{
		{0, 9, 0, 0},
		{0, 0, 0, 3},
	})
	right := matrix.New(2, 4, [][]float64{
		{9, 0, 0, 0},
		{0, 0, 3, 0},
	})

	a := MaxPool2D(Const(left), in, 2, 2).Value
	b := MaxPool2D(Const(right), in, 2, 2).Value

	if !a.Equal(b) {
		t.Errorf("a shift inside the window changed the pooled output:\n%v\nvs\n%v", a, b)
	}
}

func TestFlattenRoundTrips(t *testing.T) {
	rand.Seed(7)

	x := Param(randomMatrix(3, 4))
	flat := Flatten(x)

	if flat.Value.Rows != 12 || flat.Value.Columns != 1 {
		t.Fatalf("flattened to %dx%d, want 12x1", flat.Value.Rows, flat.Value.Columns)
	}

	weights := Const(matrix.New(12, 1, nil).Map(func(_ float64, i, _ int) float64 {
		return float64(i + 1)
	}))
	Sum(Mul(flat, weights)).Backward()

	// Each element's gradient is its own weight, reshaped back into place.
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			if got, want := x.Grad.At(i, j), float64(i*4+j+1); got != want {
				t.Errorf("grad (%d, %d) = %g, want %g", i, j, got, want)
			}
		}
	}
}

// TestConvStackLearnsShapes trains a small convolutional classifier end to
// end on synthetic images: a vertical bar, a horizontal bar, and a square.
// Position is randomized every sample, so only a translation-tolerant
// model can do it.
func TestConvStackLearnsShapes(t *testing.T) {
	rand.Seed(11)

	const (
		side    = 8
		classes = 3
		steps   = 400
	)

	in := ImageShape{Channels: 1, Height: side, Width: side}
	conv := NewConv2D(in, 4, 3, 1, 1)
	pooled := ImageShape{Channels: 4, Height: side / 2, Width: side / 2}
	head := NewLinear(pooled.Channels*pooled.Height*pooled.Width, classes)

	params := append(conv.Parameters(), head.Parameters()...)
	optimizer := Adam(0.02)

	forward := func(image matrix.Matrix) *Node {
		activated := ReLU(conv.Forward(Const(image)))
		return head.Forward(Flatten(MaxPool2D(activated, conv.Out(), 2, 2)))
	}

	for step := 0; step < steps; step++ {
		image, class := shapeSample(side)

		target := matrix.New(classes, 1, nil)
		target.Set(class, 0, 1)

		loss := SoftmaxCrossEntropy(forward(image), Const(target))
		ZeroGrad(params...)
		loss.Backward()
		optimizer.Step(params...)
	}

	correct, trials := 0, 150
	for i := 0; i < trials; i++ {
		image, class := shapeSample(side)

		logits := forward(image).Value
		best := 0
		for c := 1; c < classes; c++ {
			if logits.At(c, 0) > logits.At(best, 0) {
				best = c
			}
		}
		if best == class {
			correct++
		}
	}

	if accuracy := float64(correct) / float64(trials); accuracy < 0.9 {
		t.Errorf("shape accuracy %.3f, want the stack to learn the task", accuracy)
	}
}

// shapeSample draws one 8x8 image holding a randomly placed vertical bar,
// horizontal bar, or square.
func shapeSample(side int) (matrix.Matrix, int) {
	image := matrix.New(side, side, nil)
	class := rand.Intn(3)

	switch class {
	case 0: // vertical bar
		x := rand.Intn(side)
		for y := 0; y < side; y++ {
			image.Set(y, x, 1)
		}
	case 1: // horizontal bar
		y := rand.Intn(side)
		for x := 0; x < side; x++ {
			image.Set(y, x, 1)
		}
	case 2: // 2x2 square
		y, x := rand.Intn(side-1), rand.Intn(side-1)
		for dy := 0; dy < 2; dy++ {
			for dx := 0; dx < 2; dx++ {
				image.Set(y+dy, x+dx, 1)
			}
		}
	}

	return image, class
}

func TestConvRejectsBadShapes(t *testing.T) {
	in := ImageShape{Channels: 1, Height: 4, Width: 4}

	for name, f := range map[string]func(){
		"zero channels":  func() { NewConv2D(ImageShape{0, 4, 4}, 1, 3, 1, 0) },
		"zero filters":   func() { NewConv2D(in, 0, 3, 1, 0) },
		"oversize":       func() { NewConv2D(in, 1, 9, 1, 0) },
		"wrong input":    func() { NewConv2D(in, 1, 3, 1, 0).Forward(Const(matrix.New(3, 4, nil))) },
		"pool oversize":  func() { MaxPool2D(Const(matrix.New(4, 4, nil)), in, 9, 1) },
		"pool bad shape": func() { MaxPool2D(Const(matrix.New(3, 4, nil)), in, 2, 2) },
		"pool zero size": func() { MaxPool2D(Const(matrix.New(4, 4, nil)), in, 0, 1) },
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
