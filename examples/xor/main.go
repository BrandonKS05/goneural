package main

import (
	"fmt"
	stdrand "math/rand"
	"time"

	"github.com/BrandonKS05/goneural/goneural"
)

func init() {
	stdrand.Seed(time.Now().UnixNano())
}

func main() {
	g := goneural.New(
		0.1,
		goneural.MSE(),
		goneural.Layer{
			Nodes: 2,
		},
		goneural.Layer{
			Nodes:     4,
			Activator: goneural.Sigmoid(),
		},
		goneural.Layer{
			Nodes: 1,
		},
	)

	g.SetDebugMode(true)
	g.Train(
		goneural.SGD(),
		goneural.DataSet{
			{
				Inputs:  []float64{1, 0},
				Targets: []float64{1},
			},
			{
				Inputs:  []float64{0, 1},
				Targets: []float64{1},
			},
			{
				Inputs:  []float64{1, 1},
				Targets: []float64{0},
			},
			{
				Inputs:  []float64{0, 0},
				Targets: []float64{0},
			},
		},
		1500,
	)

	if err := g.Save("xor.goneural"); err != nil {
		panic(err)
	}

	fmt.Println("1 0 -> ", g.Predict([]float64{1, 0}), "should've been around 1")
	fmt.Println("0 1 -> ", g.Predict([]float64{0, 1}), "should've been around 1")
	fmt.Println("1 1 -> ", g.Predict([]float64{1, 1}), "should've been around 0")
	fmt.Println("0 0 -> ", g.Predict([]float64{0, 0}), "should've been around 0")
}
