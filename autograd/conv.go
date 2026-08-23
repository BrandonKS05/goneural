package autograd

import (
	"math"
	"math/rand"

	"github.com/BrandonKS05/goneural/matrix"
)

// This file adds the convolutional family to the engine. Attention and
// dense layers both treat their input as an unordered set of features --
// permute the rows of a dense layer's input and its weights permute with
// them, losing nothing. Images are not like that: a pixel means what it
// means because of what surrounds it, and the same edge is the same edge
// wherever it appears.
//
// A convolution builds both facts in. It slides one small kernel across
// every position, so a feature learned in one corner is detected in all of
// them (translation equivariance), and it looks only at a local
// neighbourhood, so the parameter count depends on the kernel size rather
// than the image size. A 5x5 kernel over a 64x64 image is 25 weights where
// a dense layer would need 4096 per output.
//
// Images are laid out as (channels * height) rows by width columns:
// channel planes stacked vertically, which keeps everything inside the
// two-dimensional Matrix the rest of the package uses. ImageShape carries
// the interpretation.

// ImageShape describes how to read a matrix as a stack of channel planes.
type ImageShape struct {
	Channels int
	Height   int
	Width    int
}

// Rows returns the row count a matrix holding this shape must have.
func (s ImageShape) Rows() int {
	return s.Channels * s.Height
}

func (s ImageShape) validate(what string) {
	if s.Channels < 1 || s.Height < 1 || s.Width < 1 {
		panic("autograd: " + what + " needs a positive image shape")
	}
}

// Conv2D is a two-dimensional convolution layer with one bias per output
// channel.
type Conv2D struct {
	// Kernels holds the filters as a matrix of shape
	// filters x (channels * size * size) -- one row per output channel,
	// each row the flattened receptive field it responds to.
	Kernels *Node
	Bias    *Node

	In     ImageShape
	Size   int
	Stride int

	// Padding is how many zero rows and columns to add on every side.
	// Padding of (Size-1)/2 with a stride of 1 keeps the output the same
	// size as the input, which is what lets convolutions stack deeply
	// without the image shrinking away.
	Padding int
}

// NewConv2D builds a convolution over the given input shape with the given
// number of filters. Weights are scaled by the He initialization, which
// accounts for a ReLU discarding half the signal by giving the weights
// twice the variance Xavier would.
func NewConv2D(in ImageShape, filters, size, stride, padding int) *Conv2D {
	in.validate("Conv2D")
	if filters < 1 || size < 1 || stride < 1 || padding < 0 {
		panic("autograd: Conv2D needs positive filters, size and stride")
	}
	if size > in.Height+2*padding || size > in.Width+2*padding {
		panic("autograd: Conv2D kernel is larger than the padded input")
	}

	fan := in.Channels * size * size
	deviation := math.Sqrt(2 / float64(fan))

	kernels := matrix.New(filters, fan, nil).Map(func(_ float64, _, _ int) float64 {
		return rand.NormFloat64() * deviation
	})

	return &Conv2D{
		Kernels: Param(kernels).Named("conv.kernels"),
		Bias:    Param(matrix.New(filters, 1, nil)).Named("conv.bias"),
		In:      in,
		Size:    size,
		Stride:  stride,
		Padding: padding,
	}
}

// Parameters returns the learnable nodes.
func (c *Conv2D) Parameters() []*Node {
	return []*Node{c.Kernels, c.Bias}
}

// Out returns the shape this layer produces.
func (c *Conv2D) Out() ImageShape {
	return ImageShape{
		Channels: c.Kernels.Value.Rows,
		Height:   (c.In.Height+2*c.Padding-c.Size)/c.Stride + 1,
		Width:    (c.In.Width+2*c.Padding-c.Size)/c.Stride + 1,
	}
}

// Forward convolves one image.
//
// The implementation is the im2col trick: rather than looping over kernel
// taps, every receptive field is copied out into a column of a patch
// matrix, and the whole convolution becomes a single matrix multiply of
// kernels against patches. That turns an awkward four-deep loop nest into
// the one operation both this package and the hardware underneath it are
// best at -- and makes the backward pass a matrix multiply too, since the
// gradient of a product is already solved.
func (c *Conv2D) Forward(x *Node) *Node {
	if x.Value.Rows != c.In.Rows() || x.Value.Columns != c.In.Width {
		panic("autograd: Conv2D input does not match the declared shape")
	}

	out := c.Out()
	patches := c.im2col(x.Value)                 // fan x (outH * outW)
	product := c.Kernels.Value.Multiply(patches) // filters x positions
	value := matrix.New(out.Rows(), out.Width, nil)

	for f := 0; f < out.Channels; f++ {
		for p := 0; p < out.Height*out.Width; p++ {
			value.Set(f*out.Height+p/out.Width, p%out.Width,
				product.At(f, p)+c.Bias.Value.At(f, 0))
		}
	}

	kernels, bias := c.Kernels, c.Bias

	return newNode(value, func(grad matrix.Matrix) {
		// Fold the (channel-plane, row, column) layout back into the
		// filters x positions matrix the multiply worked in.
		flat := matrix.New(out.Channels, out.Height*out.Width, nil)
		for f := 0; f < out.Channels; f++ {
			for p := 0; p < out.Height*out.Width; p++ {
				flat.Set(f, p, grad.At(f*out.Height+p/out.Width, p%out.Width))
			}
		}

		push(kernels, flat.Multiply(patches.Transpose()))
		push(bias, flat.RowSums())

		// Every patch column was a copy of overlapping pixels, so the
		// input's gradient is the reverse of that copy: scatter each
		// column's gradient back to the pixels it was drawn from, adding
		// where receptive fields overlapped.
		push(x, c.col2im(kernels.Value.Transpose().Multiply(flat)))
	}, x, kernels, bias)
}

// im2col lays out every receptive field as a column, in the same order
// Forward reads them back.
func (c *Conv2D) im2col(image matrix.Matrix) matrix.Matrix {
	out := c.Out()
	fan := c.In.Channels * c.Size * c.Size

	patches := matrix.New(fan, out.Height*out.Width, nil)
	for oy := 0; oy < out.Height; oy++ {
		for ox := 0; ox < out.Width; ox++ {
			column := oy*out.Width + ox

			row := 0
			for ch := 0; ch < c.In.Channels; ch++ {
				for ky := 0; ky < c.Size; ky++ {
					for kx := 0; kx < c.Size; kx++ {
						y := oy*c.Stride + ky - c.Padding
						x := ox*c.Stride + kx - c.Padding

						// Outside the image is zero, which is what padding
						// means; no explicit padded copy is needed.
						if y >= 0 && y < c.In.Height && x >= 0 && x < c.In.Width {
							patches.Set(row, column, image.At(ch*c.In.Height+y, x))
						}
						row++
					}
				}
			}
		}
	}

	return patches
}

// col2im is im2col's transpose: it accumulates a patch-shaped gradient
// back onto the image it was gathered from.
func (c *Conv2D) col2im(patches matrix.Matrix) matrix.Matrix {
	out := c.Out()
	image := matrix.New(c.In.Rows(), c.In.Width, nil)

	for oy := 0; oy < out.Height; oy++ {
		for ox := 0; ox < out.Width; ox++ {
			column := oy*out.Width + ox

			row := 0
			for ch := 0; ch < c.In.Channels; ch++ {
				for ky := 0; ky < c.Size; ky++ {
					for kx := 0; kx < c.Size; kx++ {
						y := oy*c.Stride + ky - c.Padding
						x := ox*c.Stride + kx - c.Padding

						if y >= 0 && y < c.In.Height && x >= 0 && x < c.In.Width {
							at := ch*c.In.Height + y
							image.Set(at, x, image.At(at, x)+patches.At(row, column))
						}
						row++
					}
				}
			}
		}
	}

	return image
}

// MaxPool2D downsamples each channel plane by taking the largest value in
// every window, which discards precise position while keeping whether a
// feature was present -- a small shift in the input leaves the output
// unchanged, turning the convolution's equivariance into genuine
// invariance. It has no parameters.
//
// Backward routes each window's gradient to the single pixel that won it,
// and nothing to the rest: a pixel that was not the maximum had no effect
// on the output, so it gets no credit for it.
func MaxPool2D(x *Node, in ImageShape, size, stride int) *Node {
	in.validate("MaxPool2D")
	if size < 1 || stride < 1 {
		panic("autograd: MaxPool2D needs a positive size and stride")
	}
	if x.Value.Rows != in.Rows() || x.Value.Columns != in.Width {
		panic("autograd: MaxPool2D input does not match the declared shape")
	}

	out := ImageShape{
		Channels: in.Channels,
		Height:   (in.Height-size)/stride + 1,
		Width:    (in.Width-size)/stride + 1,
	}
	if out.Height < 1 || out.Width < 1 {
		panic("autograd: MaxPool2D window is larger than the input")
	}

	value := matrix.New(out.Rows(), out.Width, nil)

	// Where each output came from, so backward can find it again.
	type source struct{ row, column int }
	winners := make([]source, out.Rows()*out.Width)

	for ch := 0; ch < in.Channels; ch++ {
		for oy := 0; oy < out.Height; oy++ {
			for ox := 0; ox < out.Width; ox++ {
				best := math.Inf(-1)
				bestRow, bestColumn := 0, 0

				for ky := 0; ky < size; ky++ {
					for kx := 0; kx < size; kx++ {
						row := ch*in.Height + oy*stride + ky
						column := ox*stride + kx

						if v := x.Value.At(row, column); v > best {
							best, bestRow, bestColumn = v, row, column
						}
					}
				}

				outRow := ch*out.Height + oy
				value.Set(outRow, ox, best)
				winners[outRow*out.Width+ox] = source{row: bestRow, column: bestColumn}
			}
		}
	}

	return newNode(value, func(grad matrix.Matrix) {
		routed := matrix.New(x.Value.Rows, x.Value.Columns, nil)
		for outRow := 0; outRow < out.Rows(); outRow++ {
			for ox := 0; ox < out.Width; ox++ {
				w := winners[outRow*out.Width+ox]
				routed.Set(w.row, w.column, routed.At(w.row, w.column)+grad.At(outRow, ox))
			}
		}
		push(x, routed)
	}, x)
}

// Flatten reshapes an image into a single column, which is how a
// convolutional stack hands off to the dense layers that classify what it
// found. Reading proceeds row by row, so the ordering is stable and the
// gradient simply reshapes back.
func Flatten(x *Node) *Node {
	value := matrix.Unflatten(x.Value.Rows*x.Value.Columns, 1, x.Value.Flatten())

	rows, columns := x.Value.Rows, x.Value.Columns

	return newNode(value, func(grad matrix.Matrix) {
		push(x, matrix.Unflatten(rows, columns, grad.Flatten()))
	}, x)
}
