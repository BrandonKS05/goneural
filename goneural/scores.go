package goneural

import (
	"math"
	"sort"

	"github.com/BrandonKS05/goneural/matrix"
)

// MeanLoss runs the network over the data set and returns the average of
// its own Loss per sample -- the same quantity Train reports per epoch, but
// measured without touching the weights. That makes it the natural way to
// watch a held-out validation set: accuracy only moves when a prediction
// crosses a decision boundary, while the loss registers the network
// growing more (or less) confident well before that. An empty data set has
// a mean loss of 0.
func (n *NeuralNetwork) MeanLoss(dataSet DataSet) float64 {
	if len(dataSet) == 0 {
		return 0
	}

	total := 0.0
	for _, ds := range dataSet {
		outputs := matrix.NewFromArray(n.Predict(ds.Inputs))
		total += n.Loss.F(outputs, matrix.NewFromArray(ds.Targets))
	}

	return total / float64(len(dataSet))
}

// TopKAccuracy is Accuracy relaxed to give credit when the true class is
// merely among the network's k highest-scoring guesses. On problems with
// many similar classes, top-1 accuracy alone cannot tell a model that is
// hopelessly lost apart from one that is consistently nearly right, and the
// gap between top-1 and top-k is exactly that distinction. k must be
// positive; a k at or past the number of classes trivially scores 1. It
// panics on single-output networks, which have no ranking to take the top
// of. An empty data set scores 0.
func (n *NeuralNetwork) TopKAccuracy(dataSet DataSet, k int) float64 {
	if k < 1 {
		panic("goneural: TopKAccuracy needs a positive k")
	}
	if n.Layers[len(n.Layers)-1].Nodes == 1 {
		panic("goneural: TopKAccuracy needs a multi-output network")
	}
	if len(dataSet) == 0 {
		return 0
	}

	correct := 0
	for _, ds := range dataSet {
		actual := classOf(ds.Targets)
		scores := n.Predict(ds.Inputs)

		// The true class is in the top k exactly when fewer than k classes
		// outscore it -- no sorting needed. Ties are resolved in the true
		// class's favour, matching ArgMax's first-one-wins rule.
		better := 0
		for i, score := range scores {
			if i != actual && score > scores[actual] {
				better++
			}
		}
		if better < k {
			correct++
		}
	}

	return float64(correct) / float64(len(dataSet))
}

// ROCAUC returns the area under the ROC curve for one class: the
// probability that a randomly chosen sample of that class is scored higher
// than a randomly chosen sample of any other. 1 is a perfect ranking, 0.5
// is chance, and below 0.5 means the scores are anti-correlated with the
// truth. Unlike Accuracy it never picks a threshold, so it measures how
// well the network *ranks* candidates and is unmoved by class imbalance --
// the reason it is the usual score for detection problems where positives
// are rare and the threshold is a downstream business decision.
//
// The score for each sample is the network's output for that class, taken
// from the single output for binary networks, where class 1 is the
// positive class and class 0 is ranked by the complement 1 - p. Those two
// binary views necessarily agree: swapping which label counts as positive
// and reversing the ranking are each a reflection of the curve, and
// together they leave the area alone. A data set with no positives or no
// negatives has no ROC curve at all, and scores 0.
func (n *NeuralNetwork) ROCAUC(dataSet DataSet, class int) float64 {
	outputs := n.Layers[len(n.Layers)-1].Nodes
	if class < 0 || (outputs == 1 && class > 1) || (outputs > 1 && class >= outputs) {
		panic("goneural: ROCAUC class out of range")
	}

	type scored struct {
		score    float64
		positive bool
	}

	samples := make([]scored, 0, len(dataSet))
	positives := 0
	for _, ds := range dataSet {
		prediction := n.Predict(ds.Inputs)

		score := prediction[0]
		if outputs > 1 {
			score = prediction[class]
		} else if class == 0 {
			score = 1 - score // rank by the complement of the positive score
		}

		positive := classOf(ds.Targets) == class
		if positive {
			positives++
		}
		samples = append(samples, scored{score: score, positive: positive})
	}

	negatives := len(samples) - positives
	if positives == 0 || negatives == 0 {
		return 0
	}

	// Computed through the Mann-Whitney U identity rather than by walking
	// the curve: the AUC equals the average rank of the positives, shifted
	// and normalized. Ties share the average of the ranks they span, which
	// is what makes a network that scores everything identically come out
	// at exactly 0.5 instead of at whichever end the sort happened to put
	// the positives.
	sort.Slice(samples, func(i, j int) bool { return samples[i].score < samples[j].score })

	rankSum := 0.0
	for i := 0; i < len(samples); {
		j := i
		for j < len(samples) && samples[j].score == samples[i].score {
			j++
		}

		// Ranks are 1-based, so this tied group spans i+1 .. j.
		averageRank := float64(i+1+j) / 2
		for k := i; k < j; k++ {
			if samples[k].positive {
				rankSum += averageRank
			}
		}
		i = j
	}

	u := rankSum - float64(positives*(positives+1))/2
	return u / float64(positives*negatives)
}

// R2Score is the coefficient of determination for regression outputs: the
// fraction of the targets' variance the network's predictions account for.
// 1 is exact, 0 means the network does no better than always predicting
// the mean of the targets, and negative values mean it does worse than
// that -- which is why it reads more usefully than a raw MSE, whose
// magnitude says nothing without knowing the scale of the data.
//
// Variance is pooled across all output dimensions, each measured against
// its own mean. A data set of fewer than two samples, or one whose targets
// never vary, has no variance to explain and scores 0.
func (n *NeuralNetwork) R2Score(dataSet DataSet) float64 {
	if len(dataSet) < 2 {
		return 0
	}

	width := len(dataSet[0].Targets)
	mean := make([]float64, width)
	for _, ds := range dataSet {
		if len(ds.Targets) != width {
			panic("goneural: R2Score needs targets of a uniform width")
		}
		for i, v := range ds.Targets {
			mean[i] += v
		}
	}
	for i := range mean {
		mean[i] /= float64(len(dataSet))
	}

	residual, total := 0.0, 0.0
	for _, ds := range dataSet {
		prediction := n.Predict(ds.Inputs)
		for i, target := range ds.Targets {
			residual += (target - prediction[i]) * (target - prediction[i])
			total += (target - mean[i]) * (target - mean[i])
		}
	}

	if total == 0 {
		return 0
	}
	return 1 - residual/total
}

// MatthewsCorrCoef reads the Matthews correlation coefficient off a
// confusion matrix: the correlation between the true and predicted
// labellings, in [-1, 1], where 1 is perfect agreement, 0 is chance, and
// -1 is perfect disagreement. It is the score to reach for when the
// classes are lopsided -- a classifier that answers "the majority class"
// every time can post a superb accuracy, but it earns an MCC of 0, because
// the coefficient only rises when every class is handled well.
//
// This is the multi-class generalization (Gorodkin, 2004), which collapses
// to the familiar 2x2 formula on a binary confusion matrix. A degenerate
// matrix -- empty, or one where a single row or column holds everything --
// has an undefined correlation and scores 0.
func MatthewsCorrCoef(confusion matrix.Matrix) float64 {
	classes := confusion.Rows
	if classes == 0 || confusion.Columns != classes {
		panic("goneural: MatthewsCorrCoef needs a square confusion matrix")
	}

	actual := make([]float64, classes)    // row sums: true occurrences
	predicted := make([]float64, classes) // column sums: predictions made
	correct, total := 0.0, 0.0

	for i := 0; i < classes; i++ {
		for j := 0; j < classes; j++ {
			v := confusion.At(i, j)
			actual[i] += v
			predicted[j] += v
			total += v
			if i == j {
				correct += v
			}
		}
	}

	if total == 0 {
		return 0
	}

	// covariance(truth, prediction) over the square roots of each side's
	// own variance -- Pearson's correlation, written for indicator vectors.
	dot := 0.0
	for k := 0; k < classes; k++ {
		dot += actual[k] * predicted[k]
	}

	sumActualSq, sumPredictedSq := 0.0, 0.0
	for k := 0; k < classes; k++ {
		sumActualSq += actual[k] * actual[k]
		sumPredictedSq += predicted[k] * predicted[k]
	}

	numerator := correct*total - dot
	denominator := math.Sqrt(total*total-sumPredictedSq) * math.Sqrt(total*total-sumActualSq)
	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}
