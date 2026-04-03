# goneural

[![Tests](https://github.com/BrandonKS05/goneural/actions/workflows/ci.yaml/badge.svg)](https://github.com/BrandonKS05/goneural/actions/workflows/ci.yaml)
[![Go Report Card](https://goreportcard.com/badge/github.com/BrandonKS05/goneural)](https://goreportcard.com/report/github.com/BrandonKS05/goneural)

**Go Report Card:** [report](https://goreportcard.com/report/github.com/BrandonKS05/goneural) — static checks for this module.

Repository: [github.com/BrandonKS05/goneural](https://github.com/BrandonKS05/goneural)

A small feedforward neural network library in Go, implemented without a heavyweight ML stack. Matrix operations use the `matrigo` module (see `go.mod`). Model save/load uses Protocol Buffers.

## Module

```
github.com/BrandonKS05/goneural
```

Import the `goneural` package (library code lives in the `goneural` subdirectory):

```go
import "github.com/BrandonKS05/goneural/goneural"
```

Pin a version (recommended):

```bash
go get github.com/BrandonKS05/goneural/goneural@v1.0.0
```

## Features

- Configurable layers, activations (sigmoid, ReLU, identity; softmax not implemented), and mean squared error loss
- Forward pass (`Predict`), training with backpropagation
- Optimizers: stochastic gradient descent, mini-batch gradient descent, batch gradient descent
- Optional genetic operators: copy, crossover, Gaussian mutation
- Serialize and deserialize networks to disk (weights, biases, layer metadata)

## Usage

```go
g := goneural.New(
	0.1,
	goneural.MSE(),
	goneural.Layer{Nodes: 2},
	goneural.Layer{Nodes: 4, Activator: goneural.Sigmoid()},
	goneural.Layer{Nodes: 1},
)

g.Train(goneural.SGD(), goneural.DataSet{
	{Inputs: []float64{1, 0}, Targets: []float64{1}},
	{Inputs: []float64{0, 1}, Targets: []float64{1}},
	{Inputs: []float64{1, 1}, Targets: []float64{0}},
	{Inputs: []float64{0, 0}, Targets: []float64{0}},
}, 5000)

g.Predict([]float64{1, 1})
```

Save and load (filename extension is up to you; `.goneural` matches the project name):

```go
g.Save("model.goneural")
g, err := goneural.Load("model.goneural")
```

## Examples

- `examples/xor` — XOR with a small MLP
- `examples/perceptron` — single perceptron demo

## Requirements

Go 1.21+ (see `go.mod`).

## Clone

```bash
git clone https://github.com/BrandonKS05/goneural.git
```

## License

See `LICENSE` in this repository.
