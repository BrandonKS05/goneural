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
- Losses: mean squared error, mean absolute error, Huber, log-cosh, categorical cross-entropy, binary cross-entropy (independent or multi-label outputs), and max-margin hinge
- Forward pass (`Predict`), training with backpropagation
- Optimizers: `SGD`, `MBGD` (mini-batch), `GD` (full batch), `MomentumSGD`/`NesterovSGD`, `Adam`, `AdamW` (decoupled weight decay), `Nadam`, `AMSGrad`, `Adamax` (decayed infinity norm), `RAdam` (variance-rectified, warmup-free), `AdaBelief` (steps on gradient surprise rather than magnitude), `LAMB` (per-layer trust ratios), `Adafactor` (rank-one factored second moments, `rows + cols` state per weight matrix instead of `rows * cols`), `Lion` (sign momentum), `RMSProp`, `AdaGrad`, `AdaDelta`, and `ConcurrentMBGD` (mini-batch with per-sample gradients computed in parallel goroutines)
- Optimizer wrappers: `Lookahead` slow/fast weight averaging, `SWA`/`EMA` tail-of-training weight averaging (`Apply` the average when training ends), and `WithWeightDecay` decoupled L2, each composable with any optimizer
- Learning-rate schedules: step, exponential and polynomial decay, cosine annealing with warm restarts, and a `WithWarmup` ramp, composable with any optimizer via `WithSchedule`/`WithScheduleFunc`
- Regularization: inverted dropout on hidden layers (`HiddenDropout`) and gradient clipping by global norm (`ClipByGlobalNorm`; the momentum optimizer wires it in through `MaxGradNorm`)
- Weight initialization: Xavier/Glorot (`InitXavier`) and He (`InitHe`)
- `LRFinder`, the learning-rate range test: ramp the rate geometrically over a throwaway copy of the network, then read the rate to train at off the resulting curve with `Suggest`
- Data preparation: `FitStandardizer`/`FitMinMaxScaler` feature scaling (fit on train, `Transform` both splits), `FitPCA` dimensionality reduction with optional whitening, `Mixup` convex-combination augmentation, and `Bootstrap` resampling
- `Ensemble` committees polled by soft (`Predict`) or hard (`Vote`) voting, trained with `Bag` bootstrap aggregation
- Metrics and helpers: `Accuracy`, `TopKAccuracy`, `MeanLoss`, `ROCAUC`, `R2Score`, `ConfusionMatrix` with `Precision`/`Recall`/`F1Score`/`MatthewsCorrCoef`, `ArgMax`, `OneHot`, `LogSumExp`, `SoftmaxWithTemperature`, data-set `Split`/`KFold`, plus a `Trainer` with optional early stopping
- Experimental: `ComplexStepGD`/`ComplexStepSGD`, an optimizer that estimates gradients via complex-step differentiation (perturbing weights by an imaginary step) instead of backprop; supports MSE loss with sigmoid/identity activations only. Also used in the test suite as an independent oracle to verify backprop's analytic gradients.
- Optional genetic operators: copy, crossover, Gaussian mutation
- Serialize and deserialize networks to disk (weights, biases, layer metadata)
- `matrix` subpackage: dense float64 matrices with the usual elementwise and product ops, row/column extraction and sums, clipping, norms, plus determinant, inverse, linear solving (Gauss-Jordan with partial pivoting), and the eigendecomposition of symmetric matrices (`SymmetricEigen`, cyclic Jacobi)

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

Pick a learning rate from the data rather than by guesswork, then train
with the averaged tail of the run:

```go
o := goneural.NewMomentumOptimizer(16, 1e-6, 0.9)
sweep := goneural.NewLRFinder(o.Optimize, func(lr float64) { o.LearningRate = lr },
	1e-6, 10, 40).Run(g, train)

if lr, ok := sweep.Suggest(0.8); ok {
	o.LearningRate = lr
}

avg := goneural.SWA(o.Optimize, 50) // ignore the first 50 epochs
g.Train(avg.Optimize, train, 200)
avg.Apply(g) // swap in the averaged weights

fmt.Println(g.Accuracy(test), g.ROCAUC(test, 1))
```

Scale the features before any of that, fitting on the training split alone:

```go
scaler := goneural.FitStandardizer(train)
train, test = scaler.Transform(train), scaler.Transform(test)

g.Predict(scaler.TransformInputs(liveInputs))
```

Or drop the input width altogether, keeping the directions the data
actually varies in — check what the reduction costs before committing to
it:

```go
pca := goneural.FitPCA(train, 10) // 10 components out of however many features
pca.Whiten = true                 // uncorrelated *and* unit-variance inputs

fmt.Println(pca.ExplainedVarianceRatio()) // e.g. [0.41 0.22 0.13 ...]
train, test = pca.Transform(train), pca.Transform(test)
```

## Examples

- `examples/mnist` — handwritten digit classification (softmax + cross-entropy + Adam, ~85-90% test accuracy; see its README for dataset download)
- `examples/xor` — XOR with a small MLP
- `examples/spiral` — two interleaved spirals, cross-validated (Nadam, dropout, warm restarts; data generated in-process, nothing to download)
- `examples/moons` — noisy two-moons classification end to end: feature scaling, a learning-rate range test, mixup, SWA, and a bagged ensemble scored by AUC and MCC
- `examples/perceptron` — single perceptron demo

## Requirements

Go 1.21+ (see `go.mod`).

## Clone

```bash
git clone https://github.com/BrandonKS05/goneural.git
```

## License

See `LICENSE` in this repository.
