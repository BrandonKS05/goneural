## Module

```
github.com/BrandonKS05/goneural
```

Import the `goneural` package (library code lives in the `goneural` subdirectory):

```go
import "github.com/BrandonKS05/goneural/goneural"
```

Install:

```bash
go get github.com/BrandonKS05/goneural/goneural@latest
```

## Features

- Configurable layers and activations: sigmoid, tanh, ReLU, leaky ReLU, ELU, SELU (self-normalizing), softplus, identity, and softmax (output layer only, paired with cross-entropy)
- Losses: mean squared error, mean absolute error, Huber, log-cosh, categorical cross-entropy
- Forward pass (`Predict`), training with backpropagation
- Optimizers: `SGD`, `MBGD` (mini-batch), `GD` (full batch), `MomentumSGD`/`NesterovSGD`, `Adam`, `AdamW` (decoupled weight decay), `Nadam`, `AMSGrad`, `Adamax` (decayed infinity norm), `RAdam` (variance-rectified, warmup-free), `AdaBelief` (steps on gradient surprise rather than magnitude), `LAMB` (per-layer trust ratios), `Adafactor` (rank-one factored second moments, `rows + cols` state per weight matrix instead of `rows * cols`), `Lion` (sign momentum), `RMSProp`, `AdaGrad`, `AdaDelta`, and `ConcurrentMBGD` (mini-batch with per-sample gradients computed in parallel goroutines)
- Optimizer wrappers: `Lookahead` slow/fast weight averaging and `WithWeightDecay` decoupled L2, each composable with any optimizer
- Learning-rate schedules: step, exponential and polynomial decay, cosine annealing with warm restarts, and a `WithWarmup` ramp, composable with any optimizer via `WithSchedule`/`WithScheduleFunc`
- Regularization: inverted dropout on hidden layers (`HiddenDropout`) and gradient clipping by global norm (`ClipByGlobalNorm`; the momentum optimizer wires it in through `MaxGradNorm`)
- Weight initialization: Xavier/Glorot (`InitXavier`) and He (`InitHe`)
- Metrics and helpers: `Accuracy`, `ConfusionMatrix` with `Precision`/`Recall`/`F1Score`, `ArgMax`, `OneHot`, `LogSumExp`, `SoftmaxWithTemperature`, data-set `Split`/`KFold`, plus a `Trainer` with optional early stopping
- Experimental: `ComplexStepGD`/`ComplexStepSGD`, an optimizer that estimates gradients via complex-step differentiation (perturbing weights by an imaginary step) instead of backprop; supports MSE loss with sigmoid/identity activations only. Also used in the test suite as an independent oracle to verify backprop's analytic gradients.
- Optional genetic operators: copy, crossover, Gaussian mutation
- Serialize and deserialize networks to disk (weights, biases, layer metadata)
- `matrix` subpackage: dense float64 matrices with the usual elementwise and product ops, row/column extraction and sums, clipping, norms, plus determinant, inverse, and linear solving (Gauss-Jordan with partial pivoting)

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

Compose the training extras — momentum with a cosine-annealed learning rate
and gradient clipping:

```go
o := goneural.NewMomentumOptimizer(16, 0.5, 0.9)
o.MaxGradNorm = 5
opt := goneural.WithScheduleFunc(o.Optimize, goneural.CosineAnnealing(0.5, 0.01, 50),
	func(lr float64) { o.LearningRate = lr })

g.InitXavier()
g.Train(opt, data, 200)
fmt.Println(g.Accuracy(data))
```

## Examples

- `examples/mnist` — handwritten digit classification (softmax + cross-entropy + Adam, ~85-90% test accuracy; see its README for dataset download)
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
