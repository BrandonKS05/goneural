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
- Data preparation: `FitStandardizer`/`FitMinMaxScaler` feature scaling (fit on train, `Transform` both splits), `Mixup` convex-combination augmentation, and `Bootstrap` resampling
- `Ensemble` committees polled by soft (`Predict`) or hard (`Vote`) voting, trained with `Bag` bootstrap aggregation
- Metrics and helpers: `Accuracy`, `TopKAccuracy`, `MeanLoss`, `ROCAUC`, `R2Score`, `ConfusionMatrix` with `Precision`/`Recall`/`F1Score`/`MatthewsCorrCoef`, `ArgMax`, `OneHot`, `LogSumExp`, `SoftmaxWithTemperature`, data-set `Split`/`KFold`, plus a `Trainer` with optional early stopping
- Experimental: `ComplexStepGD`/`ComplexStepSGD`, an optimizer that estimates gradients via complex-step differentiation (perturbing weights by an imaginary step) instead of backprop; supports MSE loss with sigmoid/identity activations only. Also used in the test suite as an independent oracle to verify backprop's analytic gradients.
- Optional genetic operators: copy, crossover, Gaussian mutation
- Serialize and deserialize networks to disk (weights, biases, layer metadata)
- `autograd` subpackage: a reverse-mode automatic differentiation engine over matrices. Build any expression out of its operations and call `Backward` on the scalar at the end; every node's gradient falls out of the chain rule rather than a hand-written recurrence. It carries `SGD`/`Adam` over graph parameters, activations the fixed architecture cannot express (`GELU` is not invertible from its output), layers (`Linear`, `LayerNorm`, `Dropout`, `Embedding`, sinusoidal `PositionalEncoding`), and all three major architecture families: `AttentionHead`/`MultiHeadAttention` with causal masking and `TransformerBlock` (pre-norm, residual); `Conv2D`/`MaxPool2D`/`Flatten` for convolutional stacks (im2col, so a convolution is one matrix multiply); and `LSTMCell` for sequences, where backpropagation through time is just the engine walking a longer graph. `MixtureOfExperts` routes each position to the top-k of several expert networks, genuinely sparsely (unrouted experts never enter the graph) with a Switch-style load-balancing loss; and `ClipGradients` rescales a whole parameter set by its global norm
- `tokenizer` subpackage: byte-pair encoding trained from a corpus — start from the 256 byte values, repeatedly fuse the most frequent adjacent pair, and common words end up as single tokens while rare ones decompose into familiar fragments. Nothing is ever out of vocabulary, so `Decode(Encode(text))` is exact for any input, emoji and invalid UTF-8 included; `CompressionRatio` reports what the vocabulary bought
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

## Examples

- `examples/mnist` — handwritten digit classification (softmax + cross-entropy + Adam, ~85-90% test accuracy; see its README for dataset download)
- `examples/xor` — XOR with a small MLP
- `examples/spiral` — two interleaved spirals, cross-validated (Nadam, dropout, warm restarts; data generated in-process, nothing to download)
- `examples/moons` — noisy two-moons classification end to end: feature scaling, a learning-rate range test, mixup, SWA, and a bagged ensemble scored by AUC and MCC
- `examples/attention` — one attention head trained on a retrieval task, printing the distribution it learned to look through (100% accuracy against 25% chance)
- `examples/tinylm` — a character-level transformer language model: embeddings, positional encoding, two causally-masked blocks, trained to a perplexity of ~1.2 in about 15 seconds, then generating text and printing its own attention map
- `examples/convnet` — a convolutional classifier on noisy shapes drawn at random positions (~95% on five classes in about a second), printing the 3x3 filters it learned and where they fired
- `examples/tokenizer` — trains a BPE vocabulary and shows what it learned: the merges, the segmentation of a sentence, and how compression falls off on unfamiliar text
- `examples/perceptron` — single perceptron demo

## Autodiff

The `goneural` package hard-codes one architecture — a stack of dense
layers whose gradient is written out by hand. The `autograd` subpackage
lifts that ceiling: operations record how they were computed, so an
expression differentiates itself.

```go
w := autograd.Param(weights)
b := autograd.Param(biases)

hidden := autograd.Tanh(autograd.AddBias(autograd.MatMul(w, autograd.Const(inputs)), b))
loss := autograd.SoftmaxCrossEntropy(hidden, autograd.Const(targets))

autograd.ZeroGrad(w, b)
loss.Backward()          // fills in w.Grad and b.Grad
optimizer.Step(w, b)
```

Every gradient in the package is verified against central finite
differences in the tests — a check that knows nothing about the chain rule
and so can only pass if the backward closures really are the derivative of
the forward code. That covers not just the individual operations but a
whole transformer block, a convolution with padding and stride, and an
LSTM unrolled over several steps — every parameter of each, at once.

The stack goes all the way up. `examples/tinylm` assembles a working
character-level language model out of these pieces:

```go
x := autograd.Add(embedding.Forward(ids), autograd.PositionalEncoding(width, len(ids)))

mask := autograd.CausalMask(len(ids))
for _, block := range blocks {
	x = block.Forward(x, mask, training)
}

logits := readout.Forward(finalNorm.Forward(x))
```

## Requirements

Go 1.21+ (see `go.mod`).

## Clone

```bash
git clone https://github.com/BrandonKS05/goneural.git
```

## License

See `LICENSE` in this repository.
