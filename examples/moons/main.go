// Command moons classifies the two-moons problem -- a pair of interleaved
// crescents that no straight line can separate -- and walks through the
// library's newer training toolkit in the order you would actually use it:
// scale the features, find a learning rate instead of guessing one, train
// with mixup augmentation under an averaged tail, score the result with
// metrics that survive class imbalance, and finally compare a single
// network against a bagged committee of five. The data is generated
// in-process, so there is nothing to download.
package main

import (
	"fmt"
	"math"
	stdrand "math/rand"

	"github.com/BrandonKS05/goneural/goneural"
)

const (
	samplesPerMoon = 300
	epochs         = 250
	members        = 5
)

// twoMoons generates n points along each of two interleaved half-circles,
// offset so the crescents nest into one another, with Gaussian jitter. The
// features are deliberately on different scales -- x spans a few units, y a
// few hundred -- so the standardizer below has something to fix.
func twoMoons(n int, noise float64) goneural.DataSet {
	data := make(goneural.DataSet, 0, 2*n)

	for moon := 0; moon < 2; moon++ {
		for i := 0; i < n; i++ {
			t := math.Pi * float64(i) / float64(n)

			x, y := math.Cos(t), math.Sin(t)
			if moon == 1 {
				x, y = 1-x, 0.5-y // the second crescent, flipped and shifted
			}

			data = append(data, goneural.DataSample{
				Inputs: []float64{
					x + stdrand.NormFloat64()*noise,
					100 * (y + stdrand.NormFloat64()*noise), // a badly scaled feature
				},
				Targets: goneural.OneHot(2, moon),
			})
		}
	}

	return data
}

func newClassifier() *goneural.NeuralNetwork {
	return goneural.New(
		0.01, // unused: the optimizer carries its own learning rate
		goneural.CrossEntropy(),
		goneural.Layer{Nodes: 2},
		goneural.Layer{Nodes: 16, Activator: goneural.Tanh()},
		goneural.Layer{Nodes: 2, Activator: goneural.Softmax()},
	)
}

// train runs one network to convergence: a fresh mixup blend every epoch,
// with the tail of the run averaged by SWA.
func train(n *goneural.NeuralNetwork, data goneural.DataSet, learningRate float64) {
	o := goneural.NewMomentumOptimizer(16, learningRate, 0.9)

	// Average the last 40% of the run, once the trajectory is circling a
	// basin rather than still descending into one.
	averaged := goneural.SWA(o.Optimize, epochs*3/5)

	for epoch := 0; epoch < epochs; epoch++ {
		// Re-drawn every epoch: the point of mixup is that the network
		// never sees the same blend twice.
		blended := data.Mixup(0.4)
		blended.Shuffle()
		averaged.Optimize(n, blended)
	}

	averaged.Apply(n)
}

func report(label string, accuracy, auc, mcc float64) {
	fmt.Printf("%-18s accuracy %.3f   AUC %.3f   MCC %.3f\n", label, accuracy, auc, mcc)
}

// weakest and strongest bracket the committee, which is the comparison
// that says whether pooling the members bought anything.
func weakest(e goneural.Ensemble, data goneural.DataSet) float64 {
	worst := 1.0
	for _, member := range e {
		worst = math.Min(worst, member.Accuracy(data))
	}
	return worst
}

func strongest(e goneural.Ensemble, data goneural.DataSet) float64 {
	best := 0.0
	for _, member := range e {
		best = math.Max(best, member.Accuracy(data))
	}
	return best
}

func main() {
	stdrand.Seed(42)

	data := twoMoons(samplesPerMoon, 0.28)
	data.Shuffle()
	rawTrain, rawTest := data.Split(0.25)

	// Fit the scaler on the training split only -- fitting on everything
	// would leak the test set's statistics into training.
	scaler := goneural.FitStandardizer(rawTrain)
	trainSet, testSet := scaler.Transform(rawTrain), scaler.Transform(rawTest)

	// Sweep the learning rate over five orders of magnitude on a throwaway
	// copy, and take the steepest point of the descent.
	probe := goneural.NewMomentumOptimizer(16, 1e-5, 0.9)
	finder := goneural.NewLRFinder(probe.Optimize,
		func(lr float64) { probe.LearningRate = lr }, 1e-5, 1, 30)
	finder.EpochsPerStep = 3

	sweep := finder.Run(newClassifier(), trainSet)

	learningRate := 0.05 // a reasonable fallback if the sweep is inconclusive
	if suggested, ok := sweep.Suggest(finder.Smoothing); ok {
		learningRate = suggested
	}
	if best, ok := sweep.Best(); ok {
		fmt.Printf("range test over %d probes: loss bottoms out at lr %.4g, training at %.4g\n\n",
			len(sweep), best.LearningRate, learningRate)
	}

	single := newClassifier()
	train(single, trainSet, learningRate)

	report("single network",
		single.Accuracy(testSet),
		single.ROCAUC(testSet, 1),
		goneural.MatthewsCorrCoef(single.ConfusionMatrix(testSet)))

	// The same recipe, five times over, each member on its own bootstrap
	// resample of the training data.
	ensemble := goneural.Bag(newClassifier(), trainSet, members,
		func(n *goneural.NeuralNetwork, d goneural.DataSet) {
			train(n, d, learningRate)
		})

	// Hard voting, for comparison with the averaged (soft) vote.
	correct := 0
	for _, ds := range testSet {
		if ensemble.Vote(ds.Inputs) == goneural.ArgMax(ds.Targets) {
			correct++
		}
	}

	fmt.Printf("%-18s accuracy %.3f   (hard vote %.3f)\n",
		fmt.Sprintf("%d-member bag", members),
		ensemble.Accuracy(testSet), float64(correct)/float64(len(testSet)))
	fmt.Printf("%-18s weakest %.3f, strongest %.3f\n",
		"  its members", weakest(ensemble, testSet), strongest(ensemble, testSet))

	fmt.Printf("\nheld-out mean loss of the single network: %.4f\n", single.MeanLoss(testSet))
}
