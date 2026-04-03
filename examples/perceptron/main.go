package main

import (
	"fmt"
	"time"

	"github.com/BrandonKS05/goneural/perceptron"
	"github.com/BrandonKS05/goneural/point"
	"github.com/BrandonKS05/goneural/rand"

	stdrand "math/rand"
)

func init() {
	stdrand.Seed(time.Now().UnixNano())
}

func main() {
	p := perceptron.New(0.1, 10000)

	// Training data
	pts := []point.Point{}
	for i := 0; i < 1000; i++ {
		pt := point.Point{
			X: rand.Float(-100, 101),
			Y: rand.Float(-100, 101),
		}
		pt.Label = point.AboveF(pt.X, pt.Y, f)
		pts = append(pts, pt)
	}

	p.Train(pts)
	fmt.Printf("%d%% correct.\n", p.Verify(f))
}

func f(x float64) float64 {
	return 3*x + 2
}
