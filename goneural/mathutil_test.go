package goneural

import (
	"math"
	"testing"

	"github.com/BrandonKS05/goneural/matrix"
)

func TestLogSumExp(t *testing.T) {
	if got, want := LogSumExp([]float64{0, 0}), math.Log(2); math.Abs(got-want) > 1e-12 {
		t.Errorf("LogSumExp([0 0]) = %g, want %g", got, want)
	}

	// Values this large overflow a naive exp-then-log; the shifted form
	// must return 1000 + ln 2 exactly.
	if got, want := LogSumExp([]float64{1000, 1000}), 1000+math.Log(2); math.Abs(got-want) > 1e-9 {
		t.Errorf("LogSumExp([1000 1000]) = %g, want %g", got, want)
	}

	if got := LogSumExp([]float64{math.Inf(-1), math.Inf(-1)}); !math.IsInf(got, -1) {
		t.Errorf("LogSumExp of all -Inf = %g, want -Inf", got)
	}

	mustPanicGoneural(t, "empty slice", func() { LogSumExp(nil) })
}

func TestSoftmaxWithTemperature(t *testing.T) {
	scores := []float64{2, 1, 0.5}

	// Temperature 1 reproduces the network's own softmax.
	got := SoftmaxWithTemperature(1, scores)
	want := softmaxVector(matrix.NewFromArray(scores)).Flatten()
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("T=1 softmax[%d] = %g, want %g", i, got[i], want[i])
		}
	}

	// Any temperature yields a probability distribution.
	for _, temp := range []float64{0.1, 1, 10} {
		sum := 0.0
		for _, p := range SoftmaxWithTemperature(temp, scores) {
			if p < 0 {
				t.Fatalf("T=%g produced a negative probability", temp)
			}
			sum += p
		}
		if math.Abs(sum-1) > 1e-12 {
			t.Fatalf("T=%g probabilities sum to %g", temp, sum)
		}
	}

	// Cooling sharpens toward the max; heating flattens toward uniform.
	cold := SoftmaxWithTemperature(0.1, scores)
	hot := SoftmaxWithTemperature(10, scores)
	base := SoftmaxWithTemperature(1, scores)
	if cold[0] <= base[0] || hot[0] >= base[0] {
		t.Errorf("temperature ordering wrong: cold %g, base %g, hot %g", cold[0], base[0], hot[0])
	}
	if math.Abs(hot[0]-1.0/3.0) > 0.05 {
		t.Errorf("hot distribution not near uniform: %v", hot)
	}

	mustPanicGoneural(t, "empty slice", func() { SoftmaxWithTemperature(1, nil) })
	mustPanicGoneural(t, "zero temperature", func() { SoftmaxWithTemperature(0, scores) })
}
