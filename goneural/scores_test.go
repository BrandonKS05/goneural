package goneural

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

// constantNetwork returns a network wired to emit a fixed output vector for
// every input, so the scores below can be checked against hand-computed
// values instead of whatever training happened to produce. A zero weight
// matrix plus a bias makes each layer's pre-activation constant.
func constantNetwork(outputs []float64) *NeuralNetwork {
	n := New(0.1, MSE(),
		Layer{Nodes: 1},
		Layer{Nodes: 1, Activator: Identity()},
		Layer{Nodes: len(outputs), Activator: Identity()},
	)

	for l := range n.Weights {
		n.Weights[l] = n.Weights[l].Scale(0)
		n.Biases[l] = n.Biases[l].Scale(0)
	}
	for i, v := range outputs {
		n.Biases[1].Set(i, 0, v)
	}

	return n
}

// scoringNetwork returns a network whose single output is its input,
// letting a test choose each sample's score directly.
func scoringNetwork() *NeuralNetwork {
	n := New(0.1, MSE(),
		Layer{Nodes: 1},
		Layer{Nodes: 1, Activator: Identity()},
		Layer{Nodes: 1, Activator: Identity()},
	)

	for l := range n.Weights {
		n.Weights[l] = n.Weights[l].Map(func(_ float64, x, y int) float64 { return 1 })
		n.Biases[l] = n.Biases[l].Scale(0)
	}

	return n
}

func TestMeanLossAveragesPerSample(t *testing.T) {
	n := constantNetwork([]float64{0.5})

	data := DataSet{
		{Inputs: []float64{0}, Targets: []float64{1.5}}, // squared error 1
		{Inputs: []float64{0}, Targets: []float64{0.5}}, // squared error 0
	}

	if got, want := n.MeanLoss(data), 0.5; math.Abs(got-want) > 1e-12 {
		t.Errorf("MeanLoss = %g, want %g", got, want)
	}
	if got := n.MeanLoss(DataSet{}); got != 0 {
		t.Errorf("MeanLoss of an empty set = %g, want 0", got)
	}
}

// TestMeanLossTracksTraining checks the held-out reading falls as the
// network learns, which is the whole point of the method.
func TestMeanLossTracksTraining(t *testing.T) {
	rand.Seed(7)

	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	data := xorData()
	before := n.MeanLoss(data)

	optimizer := Adam(2)
	for i := 0; i < 400; i++ {
		data.Shuffle()
		optimizer(n, data)
	}

	if after := n.MeanLoss(data); after >= before {
		t.Errorf("MeanLoss went from %g to %g, want a decrease", before, after)
	}
}

func TestTopKAccuracyCountsNearMisses(t *testing.T) {
	// The network always ranks class 2 first, then 1, then 0.
	n := constantNetwork([]float64{0.1, 0.3, 0.6})

	data := DataSet{
		{Inputs: []float64{0}, Targets: OneHot(3, 2)}, // rank 1
		{Inputs: []float64{0}, Targets: OneHot(3, 1)}, // rank 2
		{Inputs: []float64{0}, Targets: OneHot(3, 0)}, // rank 3
	}

	for k, want := range map[int]float64{1: 1.0 / 3, 2: 2.0 / 3, 3: 1, 5: 1} {
		if got := n.TopKAccuracy(data, k); math.Abs(got-want) > 1e-12 {
			t.Errorf("TopKAccuracy(k=%d) = %g, want %g", k, got, want)
		}
	}

	// Top-1 must agree with plain Accuracy.
	if got, want := n.TopKAccuracy(data, 1), n.Accuracy(data); got != want {
		t.Errorf("TopKAccuracy(k=1) = %g, want Accuracy %g", got, want)
	}

	if got := n.TopKAccuracy(DataSet{}, 1); got != 0 {
		t.Errorf("TopKAccuracy of an empty set = %g, want 0", got)
	}

	mustPanicGoneural(t, "non-positive k", func() { n.TopKAccuracy(data, 0) })
	mustPanicGoneural(t, "single output", func() {
		constantNetwork([]float64{0.5}).TopKAccuracy(DataSet{
			{Inputs: []float64{0}, Targets: []float64{1}},
		}, 1)
	})
}

// TestROCAUCRanksPositivesAbove pins the three landmark cases: a perfect
// ranking, a perfectly inverted one, and an all-ties one.
func TestROCAUCRanksPositivesAbove(t *testing.T) {
	n := scoringNetwork()

	labeled := func(scores []float64, positives []bool) DataSet {
		data := make(DataSet, len(scores))
		for i, s := range scores {
			target := 0.0
			if positives[i] {
				target = 1
			}
			data[i] = DataSample{Inputs: []float64{s}, Targets: []float64{target}}
		}
		return data
	}

	perfect := labeled([]float64{0.1, 0.2, 0.8, 0.9}, []bool{false, false, true, true})
	if got := n.ROCAUC(perfect, 1); math.Abs(got-1) > 1e-12 {
		t.Errorf("perfect ranking scored %g, want 1", got)
	}

	inverted := labeled([]float64{0.1, 0.2, 0.8, 0.9}, []bool{true, true, false, false})
	if got := n.ROCAUC(inverted, 1); math.Abs(got) > 1e-12 {
		t.Errorf("inverted ranking scored %g, want 0", got)
	}

	// On a binary network the two classes are two views of one ranking:
	// scoring class 0 by the complement reverses both the labels and the
	// order, which leaves the area unchanged.
	if got, want := n.ROCAUC(inverted, 0), n.ROCAUC(inverted, 1); got != want {
		t.Errorf("class 0 scored %g, want the same %g as class 1", got, want)
	}
	if got, want := n.ROCAUC(perfect, 0), n.ROCAUC(perfect, 1); got != want {
		t.Errorf("class 0 scored %g, want the same %g as class 1", got, want)
	}

	tied := labeled([]float64{0.5, 0.5, 0.5, 0.5}, []bool{true, false, true, false})
	if got := n.ROCAUC(tied, 1); math.Abs(got-0.5) > 1e-12 {
		t.Errorf("all-ties ranking scored %g, want 0.5", got)
	}

	// One positive ranked second of four: 2 of the 3 negatives sit below it.
	partial := labeled([]float64{0.1, 0.2, 0.4, 0.3}, []bool{false, false, false, true})
	if got, want := n.ROCAUC(partial, 1), 2.0/3.0; math.Abs(got-want) > 1e-12 {
		t.Errorf("partial ranking scored %g, want %g", got, want)
	}

	// No negatives at all: no curve to measure.
	allPositive := labeled([]float64{0.1, 0.9}, []bool{true, true})
	if got := n.ROCAUC(allPositive, 1); got != 0 {
		t.Errorf("single-class data scored %g, want 0", got)
	}

	mustPanicGoneural(t, "class out of range", func() { n.ROCAUC(perfect, 2) })
	mustPanicGoneural(t, "negative class", func() { n.ROCAUC(perfect, -1) })
}

// TestROCAUCIgnoresThreshold is the property that distinguishes AUC from
// accuracy: rescaling every score monotonically must not change it, even
// though it moves every prediction across the 0.5 threshold.
func TestROCAUCIgnoresThreshold(t *testing.T) {
	n := scoringNetwork()

	data := DataSet{
		{Inputs: []float64{0.9}, Targets: []float64{1}},
		{Inputs: []float64{0.7}, Targets: []float64{1}},
		{Inputs: []float64{0.6}, Targets: []float64{1}},
		{Inputs: []float64{0.55}, Targets: []float64{0}},
	}

	shrunk := make(DataSet, len(data))
	for i, ds := range data {
		shrunk[i] = DataSample{Inputs: []float64{ds.Inputs[0] / 10}, Targets: ds.Targets}
	}

	if got, want := n.ROCAUC(shrunk, 1), n.ROCAUC(data, 1); got != want {
		t.Errorf("AUC changed from %g to %g under a monotone rescaling", want, got)
	}
	if n.Accuracy(shrunk) == n.Accuracy(data) {
		t.Error("expected the rescaling to move accuracy, making the AUC check meaningful")
	}
}

func TestR2ScoreMeasuresExplainedVariance(t *testing.T) {
	targets := []float64{1, 2, 3, 4}

	data := func(n *NeuralNetwork) DataSet {
		out := make(DataSet, len(targets))
		for i, v := range targets {
			out[i] = DataSample{Inputs: []float64{v}, Targets: []float64{v}}
		}
		return out
	}

	// The identity network predicts every target exactly.
	perfect := scoringNetwork()
	if got := perfect.R2Score(data(perfect)); math.Abs(got-1) > 1e-12 {
		t.Errorf("perfect predictions scored %g, want 1", got)
	}

	// Always predicting the mean explains nothing.
	meanOnly := constantNetwork([]float64{2.5})
	if got := meanOnly.R2Score(data(meanOnly)); math.Abs(got) > 1e-12 {
		t.Errorf("mean-only predictions scored %g, want 0", got)
	}

	// A constant far from the mean is worse than useless.
	bad := constantNetwork([]float64{100})
	if got := bad.R2Score(data(bad)); got >= 0 {
		t.Errorf("wild predictions scored %g, want a negative score", got)
	}

	// Targets that never vary have no variance to explain.
	flat := DataSet{
		{Inputs: []float64{1}, Targets: []float64{7}},
		{Inputs: []float64{2}, Targets: []float64{7}},
	}
	if got := perfect.R2Score(flat); got != 0 {
		t.Errorf("constant targets scored %g, want 0", got)
	}

	if got := perfect.R2Score(DataSet{{Inputs: []float64{1}, Targets: []float64{1}}}); got != 0 {
		t.Errorf("single-sample data scored %g, want 0", got)
	}

	mustPanicGoneural(t, "ragged targets", func() {
		perfect.R2Score(DataSet{
			{Inputs: []float64{1}, Targets: []float64{1}},
			{Inputs: []float64{2}, Targets: []float64{1, 2}},
		})
	})
}

func TestMatthewsCorrCoefScoresAgreement(t *testing.T) {
	perfect := matrix.Unflatten(2, 2, []float64{5, 0, 0, 5})
	if got := MatthewsCorrCoef(perfect); math.Abs(got-1) > 1e-12 {
		t.Errorf("perfect agreement scored %g, want 1", got)
	}

	inverted := matrix.Unflatten(2, 2, []float64{0, 5, 5, 0})
	if got := MatthewsCorrCoef(inverted); math.Abs(got+1) > 1e-12 {
		t.Errorf("perfect disagreement scored %g, want -1", got)
	}

	// The lopsided-data case: 95 of 100 samples are class 0 and the
	// classifier answers 0 every time. Accuracy would read 0.95; MCC sees
	// straight through it.
	majority := matrix.Unflatten(2, 2, []float64{95, 0, 5, 0})
	if got := MatthewsCorrCoef(majority); got != 0 {
		t.Errorf("majority-class classifier scored %g, want 0", got)
	}

	// The multi-class generalization must still reach 1 on a diagonal.
	threeClass := matrix.Unflatten(3, 3, []float64{
		4, 0, 0,
		0, 3, 0,
		0, 0, 2,
	})
	if got := MatthewsCorrCoef(threeClass); math.Abs(got-1) > 1e-12 {
		t.Errorf("perfect three-class agreement scored %g, want 1", got)
	}

	// A partly-wrong three-class matrix lands strictly between chance and
	// perfect.
	muddled := matrix.Unflatten(3, 3, []float64{
		3, 1, 0,
		1, 2, 1,
		0, 1, 3,
	})
	if got := MatthewsCorrCoef(muddled); got <= 0 || got >= 1 {
		t.Errorf("muddled three-class agreement scored %g, want a value in (0, 1)", got)
	}

	if got := MatthewsCorrCoef(matrix.New(2, 2, nil)); got != 0 {
		t.Errorf("empty confusion matrix scored %g, want 0", got)
	}

	mustPanicGoneural(t, "non-square", func() { MatthewsCorrCoef(matrix.New(2, 3, nil)) })
}

// TestMatthewsCorrCoefMatchesBinaryFormula cross-checks the multi-class
// generalization against the textbook 2x2 expression.
func TestMatthewsCorrCoefMatchesBinaryFormula(t *testing.T) {
	tn, fp, fn, tp := 7.0, 3.0, 2.0, 8.0
	confusion := matrix.Unflatten(2, 2, []float64{tn, fp, fn, tp})

	want := (tp*tn - fp*fn) / math.Sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
	if got := MatthewsCorrCoef(confusion); math.Abs(got-want) > 1e-12 {
		t.Errorf("MatthewsCorrCoef = %g, want the binary formula's %g", got, want)
	}
}
