package goneural

import (
	"math/rand"
	"testing"
)

// decoyData returns a two-class problem in which only the first feature
// carries any signal: the second is noise, and the third is a constant.
func decoyData(n int) DataSet {
	data := make(DataSet, n)
	for i := range data {
		class := i % 2

		signal := -1.0
		if class == 1 {
			signal = 1
		}

		data[i] = DataSample{
			Inputs: []float64{
				signal + rand.NormFloat64()*0.3,
				rand.NormFloat64(), // pure noise
				1,                  // constant
			},
			Targets: OneHot(2, class),
		}
	}
	return data
}

// TestFeatureImportanceFindsTheSignal is the property the method exists
// for: after training, the feature that mattered must score far above the
// ones that did not.
func TestFeatureImportanceFindsTheSignal(t *testing.T) {
	rand.Seed(5)

	data := decoyData(300)

	n := New(0.1, CrossEntropy(),
		Layer{Nodes: 3},
		Layer{Nodes: 8, Activator: Tanh()},
		Layer{Nodes: 2, Activator: Softmax()},
	)
	n.Train(Adam(16), data, 150)

	importance := n.FeatureImportance(data, 5)
	if len(importance) != 3 {
		t.Fatalf("got %d importances, want one per input feature", len(importance))
	}

	if importance[0] <= 0 {
		t.Errorf("the informative feature scored %g, want a positive loss increase", importance[0])
	}
	for _, decoy := range []int{1, 2} {
		if importance[decoy] > importance[0]/4 {
			t.Errorf("uninformative feature %d scored %g against the signal's %g",
				decoy, importance[decoy], importance[0])
		}
	}

	// The same ordering should come out of the accuracy-based measure.
	accuracy := n.AccuracyImportance(data, 5)
	if accuracy[0] <= accuracy[1] || accuracy[0] <= accuracy[2] {
		t.Errorf("accuracy importances %v do not rank the signal first", accuracy)
	}
}

// TestFeatureImportanceLeavesDataAlone pins that the shuffling happens on
// copies -- a method that quietly scrambled the caller's data set would be
// a disaster to debug.
func TestFeatureImportanceLeavesDataAlone(t *testing.T) {
	rand.Seed(7)

	data := decoyData(40)

	before := make([][]float64, len(data))
	for i, ds := range data {
		before[i] = append([]float64(nil), ds.Inputs...)
	}

	n := New(0.1, CrossEntropy(),
		Layer{Nodes: 3},
		Layer{Nodes: 4, Activator: Tanh()},
		Layer{Nodes: 2, Activator: Softmax()},
	)
	n.FeatureImportance(data, 3)
	n.AccuracyImportance(data, 3)

	for i, ds := range data {
		for j, v := range ds.Inputs {
			if v != before[i][j] {
				t.Fatalf("sample %d feature %d changed from %g to %g", i, j, before[i][j], v)
			}
		}
	}
}

// TestConstantFeatureScoresZero covers the degenerate case exactly:
// shuffling a feature whose values are all identical cannot change a
// single prediction, so its importance must be exactly zero however many
// repeats are run.
func TestConstantFeatureScoresZero(t *testing.T) {
	rand.Seed(7)

	data := decoyData(60)

	n := New(0.1, CrossEntropy(),
		Layer{Nodes: 3},
		Layer{Nodes: 4, Activator: Tanh()},
		Layer{Nodes: 2, Activator: Softmax()},
	)

	if got := n.FeatureImportance(data, 4)[2]; got != 0 {
		t.Errorf("the constant feature scored %g, want exactly 0", got)
	}
	if got := n.AccuracyImportance(data, 4)[2]; got != 0 {
		t.Errorf("the constant feature scored %g on accuracy, want exactly 0", got)
	}
}

// TestUntrainedNetworkHasNoImportantFeatures checks the measure reports
// what it should on a model that learned nothing: no feature can be
// important if the network never used any of them.
func TestUntrainedNetworkHasNoImportantFeatures(t *testing.T) {
	rand.Seed(7)

	data := decoyData(200)

	n := New(0.1, CrossEntropy(),
		Layer{Nodes: 3},
		Layer{Nodes: 6, Activator: Tanh()},
		Layer{Nodes: 2, Activator: Softmax()},
	)

	trained := n.Copy()
	trained.Train(Adam(16), data, 150)

	untrained := n.FeatureImportance(data, 5)[0]
	learned := trained.FeatureImportance(data, 5)[0]

	if learned <= untrained {
		t.Errorf("the trained network scored the signal at %g, no better than the untrained %g",
			learned, untrained)
	}
}

func TestImportanceRejectsBadInput(t *testing.T) {
	n := New(0.1, MSE(),
		Layer{Nodes: 2},
		Layer{Nodes: 4, Activator: Sigmoid()},
		Layer{Nodes: 1, Activator: Sigmoid()},
	)

	mustPanicGoneural(t, "empty data set", func() { n.FeatureImportance(DataSet{}, 3) })
	mustPanicGoneural(t, "zero repeats", func() { n.FeatureImportance(xorData(), 0) })
	mustPanicGoneural(t, "empty data set", func() { n.AccuracyImportance(DataSet{}, 3) })
	mustPanicGoneural(t, "zero repeats", func() { n.AccuracyImportance(xorData(), 0) })
}
