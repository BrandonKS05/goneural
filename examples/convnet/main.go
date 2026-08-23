// Command convnet trains a small convolutional network to recognize
// shapes drawn at random positions in a noisy 12x12 image, then prints the
// filters it learned and where they fired.
//
// The point of the task is that position is randomized: the same shape
// appears anywhere in the frame, so a dense network would have to learn it
// separately for every location. A convolution learns each feature once
// and slides it everywhere, and the pooling layer then throws the exact
// position away -- which is precisely the invariance the task rewards.
//
// The whole model is assembled from the autograd package; no gradient is
// written by hand.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/BrandonKS05/goneural/autograd"
	"github.com/BrandonKS05/goneural/matrix"
)

const (
	side    = 12
	filters = 8
	steps   = 3000
	noise   = 0.03
)

var classNames = []string{"vertical", "horizontal", "square", "cross", "diagonal"}

// draw renders one shape of the given class at a random position, with a
// little salt-and-pepper noise on top.
func draw(class int) matrix.Matrix {
	image := matrix.New(side, side, nil)
	set := func(y, x int) {
		if y >= 0 && y < side && x >= 0 && x < side {
			image.Set(y, x, 1)
		}
	}

	switch class {
	case 0: // vertical bar
		x, y := rand.Intn(side), rand.Intn(side-5)
		for d := 0; d < 5; d++ {
			set(y+d, x)
		}
	case 1: // horizontal bar
		y, x := rand.Intn(side), rand.Intn(side-5)
		for d := 0; d < 5; d++ {
			set(y, x+d)
		}
	case 2: // hollow square
		y, x := rand.Intn(side-4), rand.Intn(side-4)
		for d := 0; d < 4; d++ {
			set(y, x+d)
			set(y+3, x+d)
			set(y+d, x)
			set(y+d, x+3)
		}
	case 3: // plus sign
		y, x := 2+rand.Intn(side-4), 2+rand.Intn(side-4)
		for d := -2; d <= 2; d++ {
			set(y+d, x)
			set(y, x+d)
		}
	case 4: // diagonal
		y, x := rand.Intn(side-5), rand.Intn(side-5)
		for d := 0; d < 5; d++ {
			set(y+d, x+d)
		}
	}

	return image.Map(func(v float64, _, _ int) float64 {
		if rand.Float64() < noise {
			return 1 - v // flip the occasional pixel
		}
		return v
	})
}

func sample() (matrix.Matrix, int) {
	class := rand.Intn(len(classNames))
	return draw(class), class
}

// render prints a matrix as a small ASCII picture, scaled to its own range
// so a filter's structure is visible whatever its magnitude.
func render(m matrix.Matrix, indent string) {
	shades := []string{" ", ".", ":", "+", "*", "#"}

	low, high := m.Min(), m.Max()
	span := high - low
	if span == 0 {
		span = 1
	}

	for i := 0; i < m.Rows; i++ {
		var b strings.Builder
		b.WriteString(indent)
		for j := 0; j < m.Columns; j++ {
			shade := int((m.At(i, j) - low) / span * float64(len(shades)-1))
			b.WriteString(shades[shade])
			b.WriteString(shades[shade]) // doubled, so the aspect looks square
		}
		fmt.Println(b.String())
	}
}

func main() {
	rand.Seed(7)

	in := autograd.ImageShape{Channels: 1, Height: side, Width: side}
	conv := autograd.NewConv2D(in, filters, 3, 1, 1)
	convOut := conv.Out()
	pooled := autograd.ImageShape{
		Channels: convOut.Channels,
		Height:   convOut.Height / 2,
		Width:    convOut.Width / 2,
	}
	head := autograd.NewLinear(pooled.Channels*pooled.Height*pooled.Width, len(classNames))

	params := append(conv.Parameters(), head.Parameters()...)
	optimizer := autograd.Adam(0.01)

	count := 0
	for _, p := range params {
		count += p.Value.Rows * p.Value.Columns
	}

	fmt.Printf("convolutional classifier: %d filters of 3x3, then 2x2 max pooling, %d parameters\n", filters, count)
	fmt.Printf("%d shape classes drawn at random positions in %dx%d with %.0f%% pixel noise\n\n",
		len(classNames), side, side, noise*100)

	activations := func(image matrix.Matrix) *autograd.Node {
		return autograd.ReLU(conv.Forward(autograd.Const(image)))
	}
	forward := func(image matrix.Matrix) *autograd.Node {
		return head.Forward(autograd.Flatten(
			autograd.MaxPool2D(activations(image), convOut, 2, 2)))
	}

	running := 0.0
	for step := 1; step <= steps; step++ {
		image, class := sample()

		target := matrix.New(len(classNames), 1, nil)
		target.Set(class, 0, 1)

		loss := autograd.SoftmaxCrossEntropy(forward(image), autograd.Const(target))

		autograd.ZeroGrad(params...)
		loss.Backward()
		optimizer.Step(params...)

		running += loss.Item()
		if step%500 == 0 {
			fmt.Printf("  step %3d   loss %.4f\n", step, running/500)
			running = 0
		}
	}

	// A confusion matrix over fresh samples: rows are the truth, columns
	// what the network guessed.
	const trials = 500
	confusion := matrix.New(len(classNames), len(classNames), nil)
	correct := 0

	for i := 0; i < trials; i++ {
		image, class := sample()
		predicted := argMax(forward(image).Value)

		confusion.Set(class, predicted, confusion.At(class, predicted)+1)
		if predicted == class {
			correct++
		}
	}

	fmt.Printf("\naccuracy on %d fresh images: %.3f (chance is %.2f)\n",
		trials, float64(correct)/trials, 1/float64(len(classNames)))

	fmt.Println("\nconfusion matrix (rows = truth, columns = prediction)")
	fmt.Printf("  %-12s", "")
	for _, name := range classNames {
		fmt.Printf("%9s", name[:min(6, len(name))])
	}
	fmt.Println()
	for i, name := range classNames {
		fmt.Printf("  %-12s", name)
		for j := range classNames {
			fmt.Printf("%9.0f", confusion.At(i, j))
		}
		fmt.Println()
	}

	// The filters themselves. Each is 3x3 over one input channel, so the
	// kernel row unflattens straight into a picture.
	fmt.Println("\nthe 3x3 filters it learned")
	for f := 0; f < filters; f++ {
		kernel := matrix.Unflatten(3, 3, conv.Kernels.Value.Row(f).Flatten())
		fmt.Printf("\n  filter %d (bias %+.2f)\n", f, conv.Kernels.Value.At(f, 0))
		render(kernel, "    ")
	}

	// And where the filters fire on a real image: whichever one responds
	// most strongly, since a filter that stays quiet on this sample says
	// nothing interesting about it.
	image, class := sample()
	maps := activations(image).Value

	strongest, best := 0, math.Inf(-1)
	for f := 0; f < filters; f++ {
		if total := rowsOf(maps, f, convOut.Height).Sum(); total > best {
			strongest, best = f, total
		}
	}

	fmt.Printf("\na %s sample, and what filter %d responded to\n\n", classNames[class], strongest)
	fmt.Println("  input")
	render(image, "    ")
	fmt.Printf("\n  filter %d's response\n", strongest)
	render(matrix.Unflatten(convOut.Height, convOut.Width,
		rowsOf(maps, strongest, convOut.Height).Flatten()), "    ")
}

// rowsOf extracts one channel plane from the stacked layout.
func rowsOf(m matrix.Matrix, channel, height int) matrix.Matrix {
	out := matrix.New(height, m.Columns, nil)
	for i := 0; i < height; i++ {
		for j := 0; j < m.Columns; j++ {
			out.Set(i, j, m.At(channel*height+i, j))
		}
	}
	return out
}

func argMax(m matrix.Matrix) int {
	best := 0
	for i := 1; i < m.Rows; i++ {
		if m.At(i, 0) > m.At(best, 0) {
			best = i
		}
	}
	return best
}

func min(a, b int) int {
	return int(math.Min(float64(a), float64(b)))
}
