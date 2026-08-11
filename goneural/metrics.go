package goneural

import "github.com/BrandonKS05/goneural/matrix"

// ArgMax returns the index of the largest value, taking the first one on
// ties. It panics on an empty slice, which has no largest element.
func ArgMax(values []float64) int {
	if len(values) == 0 {
		panic("goneural: ArgMax of empty slice")
	}

	best := 0
	for i, v := range values[1:] {
		if v > values[best] {
			best = i + 1
		}
	}
	return best
}

// OneHot returns a slice of length size that is all zeros except for a 1 at
// index -- the standard target encoding for classification, and the inverse
// of ArgMax. It panics if index is out of range.
func OneHot(size, index int) []float64 {
	if index < 0 || index >= size {
		panic("goneural: OneHot index out of range")
	}

	out := make([]float64, size)
	out[index] = 1
	return out
}

// classOf maps a prediction or target vector to a class index: the ArgMax
// for one-hot multi-class vectors, and a 0.5 threshold for the length-one
// vectors of binary classifiers (where the ArgMax of one value says
// nothing).
func classOf(values []float64) int {
	if len(values) == 1 {
		if values[0] >= 0.5 {
			return 1
		}
		return 0
	}
	return ArgMax(values)
}

// Accuracy runs the network over the data set and returns the fraction of
// samples classified correctly, in [0, 1]. Multi-output networks compare
// the ArgMax of the prediction against the ArgMax of the target (one-hot
// labels); single-output networks are treated as binary classifiers with
// the prediction thresholded at 0.5. An empty data set has an accuracy
// of 0.
func (n *NeuralNetwork) Accuracy(dataSet DataSet) float64 {
	if len(dataSet) == 0 {
		return 0
	}

	correct := 0
	for _, ds := range dataSet {
		if classOf(n.Predict(ds.Inputs)) == classOf(ds.Targets) {
			correct++
		}
	}

	return float64(correct) / float64(len(dataSet))
}

// ConfusionMatrix runs the network over the data set and returns a k x k
// matrix whose entry (i, j) counts the samples of true class i that the
// network predicted as class j -- rows are the truth, columns the
// prediction, so a perfect classifier fills only the diagonal. Classes are
// derived the same way Accuracy derives them; single-output binary
// networks yield a 2x2 matrix.
func (n *NeuralNetwork) ConfusionMatrix(dataSet DataSet) matrix.Matrix {
	classes := n.Layers[len(n.Layers)-1].Nodes
	if classes == 1 {
		classes = 2
	}

	confusion := matrix.New(classes, classes, nil)
	for _, ds := range dataSet {
		actual := classOf(ds.Targets)
		predicted := classOf(n.Predict(ds.Inputs))
		confusion.Set(actual, predicted, confusion.At(actual, predicted)+1)
	}

	return confusion
}

// Precision reads a class's precision from a confusion matrix: of
// everything predicted as that class (its column), the fraction that truly
// was. A class never predicted has a precision of 0.
func Precision(confusion matrix.Matrix, class int) float64 {
	predicted := 0.0
	for i := 0; i < confusion.Rows; i++ {
		predicted += confusion.At(i, class)
	}
	if predicted == 0 {
		return 0
	}
	return confusion.At(class, class) / predicted
}

// Recall reads a class's recall from a confusion matrix: of everything
// truly in that class (its row), the fraction the classifier found. A
// class with no true samples has a recall of 0.
func Recall(confusion matrix.Matrix, class int) float64 {
	actual := 0.0
	for j := 0; j < confusion.Columns; j++ {
		actual += confusion.At(class, j)
	}
	if actual == 0 {
		return 0
	}
	return confusion.At(class, class) / actual
}

// F1Score is the harmonic mean of a class's precision and recall, which
// only rewards classifiers that are good at both. It is 0 when either
// component is 0.
func F1Score(confusion matrix.Matrix, class int) float64 {
	p := Precision(confusion, class)
	r := Recall(confusion, class)
	if p+r == 0 {
		return 0
	}
	return 2 * p * r / (p + r)
}
