// Command spiral trains a classifier on the classic two-interleaved-spirals
// problem -- a data set no linear model can separate -- and evaluates it
// with k-fold cross-validation. The data is generated in-process, so unlike
// the mnist example there is nothing to download. It exercises the newer
// training toolkit end to end: Nadam, hidden-layer dropout, a warm-restart
// cosine learning-rate schedule, and the confusion-matrix metrics.
package main

import (
	"fmt"
	"math"
	stdrand "math/rand"

	"github.com/BrandonKS05/goneural/goneural"
)

const (
	samplesPerSpiral = 100
	folds            = 5
	epochs           = 300
)

// twoSpirals generates n points along each of two interleaved spiral arms,
// jittered with a little Gaussian noise, labeled one-hot by arm.
func twoSpirals(n int, noise float64) goneural.DataSet {
	data := make(goneural.DataSet, 0, 2*n)

	for arm := 0; arm < 2; arm++ {
		phase := float64(arm) * math.Pi
		for i := 0; i < n; i++ {
			t := 0.25 + 2.25*float64(i)/float64(n) // radians along the arm
			r := t / 4                            // radius grows with angle

			x := r*math.Cos(t*math.Pi+phase) + stdrand.NormFloat64()*noise
			y := r*math.Sin(t*math.Pi+phase) + stdrand.NormFloat64()*noise

			data = append(data, goneural.DataSample{
				Inputs:  []float64{x, y},
				Targets: goneural.OneHot(2, arm),
			})
		}
	}

	return data
}

func newClassifier() *goneural.NeuralNetwork {
	n := goneural.New(
		0.01, // unused: Nadam carries its own learning rate
		goneural.CrossEntropy(),
		goneural.Layer{Nodes: 2},
		goneural.Layer{Nodes: 24, Activator: goneural.Tanh()},
		goneural.Layer{Nodes: 24, Activator: goneural.Tanh()},
		goneural.Layer{Nodes: 2, Activator: goneural.Softmax()},
	)

	// Deliberately NOT InitXavier() here, and it's worth knowing why: the
	// two-spiral set is point-symmetric (arm 1 is arm 0 negated), tanh is an
	// odd function, and Xavier starts all biases at zero -- which makes the
	// whole network an odd function of its input, started deep in tanh's
	// linear regime. That symmetric starting point is a trap this problem
	// never escapes: training flatlines at chance accuracy. The default
	// initialization's random biases break the symmetry from step one.
	n.HiddenDropout = 0.01
	return n
}

func main() {
	stdrand.Seed(42)

	data := twoSpirals(samplesPerSpiral, 0.02)
	data.Shuffle()

	totalAccuracy := 0.0
	for i, fold := range data.KFold(folds) {
		n := newClassifier()

		// Nadam under a cosine schedule with warm restarts every 200 epochs.
		o := goneural.NewNadamOptimizer(8, 0.01)
		optimizer := goneural.WithScheduleFunc(o.Optimize,
			goneural.CosineAnnealing(0.01, 0.001, 200),
			func(lr float64) { o.LearningRate = lr })

		n.Train(optimizer, fold.Train, epochs)

		accuracy := n.Accuracy(fold.Test)
		totalAccuracy += accuracy
		fmt.Printf("fold %d/%d: accuracy %.3f\n", i+1, folds, accuracy)

		if i == folds-1 {
			confusion := n.ConfusionMatrix(fold.Test)
			fmt.Printf("\nlast fold's confusion matrix (rows = truth):\n%v", confusion)
			fmt.Printf("arm 0: precision %.3f recall %.3f F1 %.3f\n",
				goneural.Precision(confusion, 0), goneural.Recall(confusion, 0), goneural.F1Score(confusion, 0))
			fmt.Printf("arm 1: precision %.3f recall %.3f F1 %.3f\n",
				goneural.Precision(confusion, 1), goneural.Recall(confusion, 1), goneural.F1Score(confusion, 1))
		}
	}

	fmt.Printf("\nmean cross-validated accuracy over %d folds: %.3f\n", folds, totalAccuracy/folds)
}
