package goneural

import "fmt"

// Split partitions the data set into a training and a test portion, with
// testFraction (in the open interval (0, 1)) of the samples -- at least
// one, rounded to nearest -- going to the test set. The split is
// positional: the tail of the set becomes the test data, so call Shuffle
// first unless the order is already random. Like Batch, the returned sets
// are views sharing the receiver's backing array, not copies. It panics if
// the set has fewer than two samples, which cannot be split.
func (t DataSet) Split(testFraction float64) (train, test DataSet) {
	if testFraction <= 0 || testFraction >= 1 {
		panic(fmt.Sprintf("goneural: Split testFraction %g outside (0, 1)", testFraction))
	}
	if len(t) < 2 {
		panic("goneural: Split needs at least two samples")
	}

	testLen := int(float64(len(t))*testFraction + 0.5)
	if testLen < 1 {
		testLen = 1
	}
	if testLen > len(t)-1 {
		testLen = len(t) - 1
	}

	cut := len(t) - testLen
	return t[:cut], t[cut:]
}

// Fold is one train/test partition produced by KFold.
type Fold struct {
	Train DataSet
	Test  DataSet
}

// KFold partitions the data set into k folds for cross-validation: fold i
// holds the i-th contiguous chunk as its test set and everything else as
// its training set, so across all folds every sample appears in exactly
// one test set. Chunk boundaries distribute any remainder evenly, and the
// partition is positional -- Shuffle first unless the order is already
// random. Test sets are views into the receiver; training sets are fresh
// copies, since their two halves aren't contiguous. It panics unless
// 2 <= k <= len(t).
func (t DataSet) KFold(k int) []Fold {
	if k < 2 || k > len(t) {
		panic(fmt.Sprintf("goneural: KFold with k=%d for %d samples", k, len(t)))
	}

	folds := make([]Fold, k)
	for i := 0; i < k; i++ {
		start := i * len(t) / k
		end := (i + 1) * len(t) / k

		train := make(DataSet, 0, len(t)-(end-start))
		train = append(train, t[:start]...)
		train = append(train, t[end:]...)

		folds[i] = Fold{Train: train, Test: t[start:end]}
	}

	return folds
}
