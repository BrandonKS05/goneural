package goneural

import "testing"

func numberedDataSet(n int) DataSet {
	ds := make(DataSet, n)
	for i := range ds {
		ds[i] = DataSample{Inputs: []float64{float64(i)}, Targets: []float64{0}}
	}
	return ds
}

func TestSplit(t *testing.T) {
	ds := numberedDataSet(10)

	train, test := ds.Split(0.3)
	if len(train) != 7 || len(test) != 3 {
		t.Fatalf("Split(0.3) sizes = %d/%d, want 7/3", len(train), len(test))
	}

	// The two views partition the original: tail goes to test.
	if train[6].Inputs[0] != 6 || test[0].Inputs[0] != 7 {
		t.Errorf("Split cut in the wrong place: train ends %g, test starts %g", train[6].Inputs[0], test[0].Inputs[0])
	}

	// Tiny fractions still put at least one sample on each side.
	train, test = ds.Split(0.01)
	if len(test) != 1 || len(train) != 9 {
		t.Errorf("Split(0.01) sizes = %d/%d, want 9/1", len(train), len(test))
	}

	mustPanicGoneural(t, "fraction 0", func() { ds.Split(0) })
	mustPanicGoneural(t, "fraction 1", func() { ds.Split(1) })
	mustPanicGoneural(t, "too few samples", func() { numberedDataSet(1).Split(0.5) })
}

func TestKFold(t *testing.T) {
	// 10 samples across 3 folds: sizes must come out 3/3/4-ish (evenly
	// distributed remainder) and every sample lands in exactly one test set.
	ds := numberedDataSet(10)
	folds := ds.KFold(3)

	if len(folds) != 3 {
		t.Fatalf("KFold(3) returned %d folds", len(folds))
	}

	seen := make(map[float64]int)
	for i, fold := range folds {
		if len(fold.Train)+len(fold.Test) != len(ds) {
			t.Errorf("fold %d: train %d + test %d != %d", i, len(fold.Train), len(fold.Test), len(ds))
		}
		for _, s := range fold.Test {
			seen[s.Inputs[0]]++
		}

		// No sample may sit in both halves of the same fold.
		inTest := make(map[float64]bool)
		for _, s := range fold.Test {
			inTest[s.Inputs[0]] = true
		}
		for _, s := range fold.Train {
			if inTest[s.Inputs[0]] {
				t.Errorf("fold %d: sample %g in both train and test", i, s.Inputs[0])
			}
		}
	}

	if len(seen) != len(ds) {
		t.Errorf("test folds cover %d distinct samples, want %d", len(seen), len(ds))
	}
	for v, count := range seen {
		if count != 1 {
			t.Errorf("sample %g appeared in %d test folds, want exactly 1", v, count)
		}
	}

	mustPanicGoneural(t, "k too small", func() { ds.KFold(1) })
	mustPanicGoneural(t, "k exceeds samples", func() { ds.KFold(11) })
}
