package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	stdrand "math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/BrandonKS05/goneural/goneural"
)

const (
	trainSamples = 4000
	testSamples  = 1000
	epochs       = 10
	imageSize    = 28 * 28
	numClasses   = 10
)

func main() {
	stdrand.Seed(time.Now().UnixNano())

	dataDir := "data"
	if len(os.Args) > 1 {
		dataDir = os.Args[1]
	}

	trainImages, err := readImages(filepath.Join(dataDir, "train-images-idx3-ubyte.gz"))
	must(err)
	trainLabels, err := readLabels(filepath.Join(dataDir, "train-labels-idx1-ubyte.gz"))
	must(err)
	testImages, err := readImages(filepath.Join(dataDir, "t10k-images-idx3-ubyte.gz"))
	must(err)
	testLabels, err := readLabels(filepath.Join(dataDir, "t10k-labels-idx1-ubyte.gz"))
	must(err)

	train := toDataSet(trainImages, trainLabels)
	train.Shuffle()
	train = train[:trainSamples]

	test := toDataSet(testImages, testLabels)
	test.Shuffle()
	test = test[:testSamples]

	n := goneural.New(
		0.001,
		goneural.CrossEntropy(),
		goneural.Layer{Nodes: imageSize},
		goneural.Layer{Nodes: 64, Activator: goneural.Sigmoid()},
		goneural.Layer{Nodes: numClasses, Activator: goneural.Softmax()},
	)

	optimizer := goneural.Adam(32)
	log.Printf("training on %d samples for %d epochs...", len(train), epochs)
	start := time.Now()
	for epoch := 1; epoch <= epochs; epoch++ {
		train.Shuffle()
		loss := optimizer(n, train)
		log.Printf("epoch %d/%d: avg loss %.4f, test accuracy %.1f%%",
			epoch, epochs, loss/float64(len(train)), accuracy(n, test)*100)
	}
	log.Printf("trained in %s", time.Since(start))

	fmt.Printf("final test accuracy: %.1f%% (%d test samples)\n", accuracy(n, test)*100, len(test))
}

func accuracy(n *goneural.NeuralNetwork, ds goneural.DataSet) float64 {
	correct := 0
	for _, sample := range ds {
		if argmax(n.Predict(sample.Inputs)) == argmax(sample.Targets) {
			correct++
		}
	}
	return float64(correct) / float64(len(ds))
}

func argmax(v []float64) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

func toDataSet(images [][]float64, labels []byte) goneural.DataSet {
	ds := make(goneural.DataSet, len(images))
	for i := range images {
		targets := make([]float64, numClasses)
		targets[labels[i]] = 1 // one-hot encoding
		ds[i] = goneural.DataSample{Inputs: images[i], Targets: targets}
	}
	return ds
}

// readImages parses the IDX3 format: magic(4) count(4) rows(4) cols(4)
// followed by count*rows*cols unsigned bytes. Pixels are scaled to [0, 1].
func readImages(path string) ([][]float64, error) {
	r, close, err := openGzip(path)
	if err != nil {
		return nil, err
	}
	defer close()

	var header struct{ Magic, Count, Rows, Cols int32 }
	if err := binary.Read(r, binary.BigEndian, &header); err != nil {
		return nil, err
	}
	if header.Magic != 2051 {
		return nil, fmt.Errorf("%s: bad magic %d, want 2051", path, header.Magic)
	}

	pixels := make([]byte, int(header.Count)*int(header.Rows)*int(header.Cols))
	if _, err := io.ReadFull(r, pixels); err != nil {
		return nil, err
	}

	size := int(header.Rows) * int(header.Cols)
	images := make([][]float64, header.Count)
	for i := range images {
		img := make([]float64, size)
		for j, p := range pixels[i*size : (i+1)*size] {
			img[j] = float64(p) / 255
		}
		images[i] = img
	}

	return images, nil
}

// readLabels parses the IDX1 format: magic(4) count(4) followed by count
// unsigned bytes, each a digit 0-9.
func readLabels(path string) ([]byte, error) {
	r, close, err := openGzip(path)
	if err != nil {
		return nil, err
	}
	defer close()

	var header struct{ Magic, Count int32 }
	if err := binary.Read(r, binary.BigEndian, &header); err != nil {
		return nil, err
	}
	if header.Magic != 2049 {
		return nil, fmt.Errorf("%s: bad magic %d, want 2049", path, header.Magic)
	}

	labels := make([]byte, header.Count)
	if _, err := io.ReadFull(r, labels); err != nil {
		return nil, err
	}

	return labels, nil
}

func openGzip(path string) (io.Reader, func(), error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("%w (download the MNIST files first, see examples/mnist/README.md)", err)
	}

	gz, err := gzip.NewReader(f)
	if err != nil {
		f.Close()
		return nil, nil, err
	}

	return gz, func() { gz.Close(); f.Close() }, nil
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
