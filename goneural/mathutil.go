package goneural

import "math"

// LogSumExp computes log(exp(x_1) + ... + exp(x_n)) without overflowing:
// the largest value is factored out before exponentiating, so inputs in
// the hundreds -- which would push math.Exp past float64 range -- come
// through exactly. It is the standard building block for working with
// log-probabilities. It panics on an empty slice.
func LogSumExp(values []float64) float64 {
	if len(values) == 0 {
		panic("goneural: LogSumExp of empty slice")
	}

	max := values[0]
	for _, v := range values[1:] {
		if v > max {
			max = v
		}
	}
	// A slice of all -Inf sums to zero probability; return -Inf rather
	// than letting the shift produce NaN from Inf - Inf.
	if math.IsInf(max, -1) {
		return math.Inf(-1)
	}

	sum := 0.0
	for _, v := range values {
		sum += math.Exp(v - max)
	}
	return max + math.Log(sum)
}

// SoftmaxWithTemperature turns raw scores into a probability distribution
// with a temperature knob: dividing the scores by temperature before the
// (numerically stable) softmax sharpens the distribution when temperature
// is below 1 and flattens it toward uniform when above 1. A temperature of
// exactly 1 reproduces the softmax the network's output layer applies. It
// panics on an empty slice or a non-positive temperature.
func SoftmaxWithTemperature(temperature float64, values []float64) []float64 {
	if len(values) == 0 {
		panic("goneural: SoftmaxWithTemperature of empty slice")
	}
	if temperature <= 0 {
		panic("goneural: SoftmaxWithTemperature needs a positive temperature")
	}

	scaled := make([]float64, len(values))
	for i, v := range values {
		scaled[i] = v / temperature
	}

	norm := LogSumExp(scaled)
	out := make([]float64, len(scaled))
	for i, v := range scaled {
		out[i] = math.Exp(v - norm)
	}
	return out
}
