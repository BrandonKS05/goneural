package goneural

import (
	"math"

	"github.com/BrandonKS05/goneural/matrix"
)

// Loss is a loss function
// it contains the normal f(x) and the derivative f'(x)
// By convention (matching every call site in this package), F and FPrime
// are called as F(prediction, target) -- despite the y/yHat parameter
// names, the first argument is always the network's output and the second
// is always the ground truth.
type Loss struct {
	Name   lossName
	F      func(y, yHat matrix.Matrix) float64
	FPrime func(y, yHat matrix.Matrix) matrix.Matrix
}

type lossName string

const (
	mse           lossName = "mse"
	crossEntropy  lossName = "cross_entropy"
	minLogProb             = 1e-12 // clamps log(0)/divide-by-0 for predictions right at the boundary
)

func getLossFromname(a lossName) Loss {
	switch a {
	case mse:
		return MSE()
	case crossEntropy:
		return CrossEntropy()
	default:
		return MSE()
	}
}

// MSE is the Mean Squared Error
func MSE() Loss {
	return Loss{
		Name: mse,
		F: func(y, yHat matrix.Matrix) float64 {
			out := yHat.SubtractMatrix(y)
			out = out.HadamardProduct(out)
			return out.Sum() / float64(y.Rows)
		},
		FPrime: func(y, yHat matrix.Matrix) matrix.Matrix {
			return yHat.SubtractMatrix(y).Divide(float64(y.Rows * y.Columns))
		},
	}
}

// CrossEntropy is categorical cross-entropy, the standard loss for
// multi-class classification. It's normally paired with a Softmax output
// layer -- MBGD/Adam/ConcurrentMBGD special-case that pairing with the
// well-known combined gradient shortcut (prediction - target) rather than
// going through FPrime below, since deriving the full softmax Jacobian
// generically isn't worth it for this library. FPrime is still provided so
// the Loss is correct standalone (e.g. if ever paired with a different
// output activation).
func CrossEntropy() Loss {
	return Loss{
		Name: crossEntropy,
		F: func(y, yHat matrix.Matrix) float64 {
			pred := y.Flatten()
			truth := yHat.Flatten()

			sum := 0.0
			for i, p := range pred {
				sum -= truth[i] * math.Log(math.Max(p, minLogProb))
			}
			return sum
		},
		FPrime: func(y, yHat matrix.Matrix) matrix.Matrix {
			pred := y.Flatten()
			truth := yHat.Flatten()

			out := make([]float64, len(pred))
			for i, p := range pred {
				out[i] = -truth[i] / math.Max(p, minLogProb)
			}
			return matrix.Unflatten(y.Rows, y.Columns, out)
		},
	}
}
