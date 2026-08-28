package goneural

import "math/rand"

// FeatureImportance measures how much the network relies on each input
// feature, by breaking one feature at a time and watching what happens to
// the loss.
//
// The trick is permutation rather than deletion: a feature's values are
// shuffled across the data set, which destroys its relationship to the
// target while leaving its distribution intact. A feature the network
// leans on takes the loss up sharply when scrambled; one it ignores
// changes nothing. Removing the feature and retraining would answer a
// different (and more expensive) question -- whether the information is
// available elsewhere -- while this answers what *this* network actually
// uses.
//
// It is model-agnostic by construction: nothing here inspects a weight.
// That is what makes it trustworthy where reading weights is not -- a
// large weight on a feature the network cancels out downstream looks
// important and is not.
//
// Returned values are the mean increase in loss over `repeats` shuffles,
// one per input feature, in input order. Larger is more important;
// slightly negative simply means noise, and should be read as zero. Use at
// least a few repeats: a single shuffle can flatter or damn a feature by
// luck. It panics on an empty data set or a non-positive repeat count.
func (n *NeuralNetwork) FeatureImportance(dataSet DataSet, repeats int) []float64 {
	if len(dataSet) == 0 {
		panic("goneural: FeatureImportance needs a non-empty data set")
	}
	if repeats < 1 {
		panic("goneural: FeatureImportance needs at least one repeat")
	}

	features := n.Layers[0].Nodes
	baseline := n.MeanLoss(dataSet)
	importance := make([]float64, features)

	// One reusable scratch copy: the samples' input slices are replaced
	// rather than written through, so the caller's data is never touched.
	shuffled := make(DataSet, len(dataSet))

	for feature := 0; feature < features; feature++ {
		for repeat := 0; repeat < repeats; repeat++ {
			order := rand.Perm(len(dataSet))

			for i, ds := range dataSet {
				inputs := append([]float64(nil), ds.Inputs...)
				inputs[feature] = dataSet[order[i]].Inputs[feature]
				shuffled[i] = DataSample{Inputs: inputs, Targets: ds.Targets}
			}

			importance[feature] += n.MeanLoss(shuffled) - baseline
		}

		importance[feature] /= float64(repeats)
	}

	return importance
}

// AccuracyImportance is FeatureImportance measured in accuracy lost rather
// than loss gained: the drop in correct classifications when a feature is
// scrambled. It reads more directly for a classifier -- "this feature is
// worth 12 points of accuracy" -- but it is blunter, since a prediction
// has to cross a decision boundary to register at all, and a feature the
// network merely leans on for confidence will score zero here while
// showing up clearly in the loss.
func (n *NeuralNetwork) AccuracyImportance(dataSet DataSet, repeats int) []float64 {
	if len(dataSet) == 0 {
		panic("goneural: AccuracyImportance needs a non-empty data set")
	}
	if repeats < 1 {
		panic("goneural: AccuracyImportance needs at least one repeat")
	}

	features := n.Layers[0].Nodes
	baseline := n.Accuracy(dataSet)
	importance := make([]float64, features)

	shuffled := make(DataSet, len(dataSet))

	for feature := 0; feature < features; feature++ {
		for repeat := 0; repeat < repeats; repeat++ {
			order := rand.Perm(len(dataSet))

			for i, ds := range dataSet {
				inputs := append([]float64(nil), ds.Inputs...)
				inputs[feature] = dataSet[order[i]].Inputs[feature]
				shuffled[i] = DataSample{Inputs: inputs, Targets: ds.Targets}
			}

			importance[feature] += baseline - n.Accuracy(shuffled)
		}

		importance[feature] /= float64(repeats)
	}

	return importance
}
