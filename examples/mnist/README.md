# MNIST example

Trains a 784-64-10 network (sigmoid hidden layer, softmax output,
cross-entropy loss, Adam optimizer) on a subset of the MNIST handwritten
digit dataset and reports test-set accuracy.

## Getting the data

Download the four dataset files into `examples/mnist/data/` (they are
gitignored). Any MNIST mirror works; the one below is the same S3 bucket
PyTorch's torchvision uses:

```bash
mkdir -p data
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -sSL -o "data/$f" "https://ossci-datasets.s3.amazonaws.com/mnist/$f"
done
```

## Running

```bash
go run . [path-to-data-dir]
```

The data directory defaults to `./data`. Training uses a 4000-image subset
(the library is pure Go with no BLAS/GPU acceleration, so full-dataset
training is slow) and typically reaches ~85-90% test accuracy in about half
a minute.
