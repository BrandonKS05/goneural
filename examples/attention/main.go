// Command attention trains a single scaled dot-product attention head on a
// retrieval task and then prints what it learned to look at.
//
// The task: three content positions each carry a random symbol, exactly one
// of them is flagged, and a fourth query position must report the flagged
// symbol. Which position holds the answer changes every sample, so no fixed
// set of weights can solve it -- the model has to decide where to look from
// the content itself. That is precisely what attention does and what the
// dense stacks in the goneural package cannot express.
//
// Everything below is built from the autograd package's ordinary
// operations. No gradient is written by hand anywhere.
package main

import (
	"fmt"
	"math/rand"
	"strings"

	"github.com/BrandonKS05/goneural/autograd"
	"github.com/BrandonKS05/goneural/matrix"
)

const (
	symbols   = 4            // how many distinct symbols exist
	contents  = 3            // content positions per sequence
	positions = contents + 1 // plus the query position
	model     = symbols + 2  // one-hot symbol, a flag row, a query row
	headWidth = 8
	steps     = 1500
)

var symbolNames = []string{"A", "B", "C", "D"}

// sample builds one sequence: random symbols, one flagged at random, and a
// final query position. The answer is the flagged symbol.
func sample() (input matrix.Matrix, flagged, answer int) {
	input = matrix.New(model, positions, nil)
	flagged = rand.Intn(contents)

	for j := 0; j < contents; j++ {
		symbol := rand.Intn(symbols)
		input.Set(symbol, j, 1)
		if j == flagged {
			input.Set(symbols, j, 1) // the flag
			answer = symbol
		}
	}
	input.Set(symbols+1, positions-1, 1) // the query marker

	return input, flagged, answer
}

func describe(input matrix.Matrix, flagged int) string {
	var b strings.Builder
	for j := 0; j < contents; j++ {
		for s := 0; s < symbols; s++ {
			if input.At(s, j) == 1 {
				b.WriteString(symbolNames[s])
			}
		}
		if j == flagged {
			b.WriteString("*") // the flagged one
		} else {
			b.WriteString(" ")
		}
		b.WriteString(" ")
	}
	b.WriteString("| ?")
	return b.String()
}

// bar renders a weight in [0, 1] as a short text meter, so the attention
// pattern is legible at a glance.
func bar(weight float64) string {
	const width = 24
	filled := int(weight*width + 0.5)
	return "[" + strings.Repeat("#", filled) + strings.Repeat(".", width-filled) + "]"
}

func main() {
	rand.Seed(42)

	head := autograd.NewAttentionHead(model, headWidth)

	// A fixed one-hot selector reads the query position's output column.
	selector := matrix.New(positions, 1, nil)
	selector.Set(positions-1, 0, 1)

	readout := autograd.Param(matrix.New(symbols, model, nil).
		Map(func(_ float64, _, _ int) float64 { return rand.NormFloat64() * 0.3 })).
		Named("readout")

	params := append(head.Parameters(), readout)
	optimizer := autograd.Adam(0.05)

	predict := func(input matrix.Matrix) *autograd.Node {
		attended := head.Forward(autograd.Const(input), nil)
		return autograd.MatMul(readout, autograd.MatMul(attended, autograd.Const(selector)))
	}

	fmt.Printf("training one attention head (%d model dims, %d-wide head) for %d steps\n\n",
		model, headWidth, steps)

	running := 0.0
	for step := 1; step <= steps; step++ {
		input, _, answer := sample()

		target := matrix.New(symbols, 1, nil)
		target.Set(answer, 0, 1)

		loss := autograd.SoftmaxCrossEntropy(predict(input), autograd.Const(target))

		autograd.ZeroGrad(params...)
		loss.Backward()
		optimizer.Step(params...)

		running += loss.Item()
		if step%300 == 0 {
			fmt.Printf("  step %4d   loss %.4f\n", step, running/300)
			running = 0
		}
	}

	correct := 0
	const trials = 500
	for i := 0; i < trials; i++ {
		input, _, answer := sample()
		if argMax(predict(input).Value) == answer {
			correct++
		}
	}
	fmt.Printf("\nretrieval accuracy over %d fresh sequences: %.3f (chance is %.3f)\n",
		trials, float64(correct)/trials, 1.0/symbols)

	// The interesting part: where does the query position actually look?
	fmt.Println("\nwhat the query attends to, on three fresh sequences")
	fmt.Println("(sequence shown as symbols, * marks the flagged one)")

	for i := 0; i < 3; i++ {
		input, flagged, answer := sample()
		weights := head.Weights(autograd.Const(input), nil)

		fmt.Printf("\n  %s   -> predicted %s, answer %s\n",
			describe(input, flagged),
			symbolNames[argMax(predict(input).Value)],
			symbolNames[answer])

		for key := 0; key < positions; key++ {
			label := fmt.Sprintf("position %d", key)
			if key == positions-1 {
				label = "query pos"
			}
			if key == flagged {
				label += " *"
			}

			w := weights.At(key, positions-1) // the query's weight on this key
			fmt.Printf("    %-13s %s %.3f\n", label, bar(w), w)
		}
	}
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
