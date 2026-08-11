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
	mse          lossName = "mse"
	crossEntropy lossName = "cross_entropy"
	mae          lossName = "mae"
	huber        lossName = "huber"
	minLogProb            = 1e-12 // clamps log(0)/divide-by-0 for predictions right at the boundary
)

func getLossFromname(a lossName) Loss {
	switch a {
	case mse:
		return MSE()
	case crossEntropy:
		return CrossEntropy()
	case mae:
		return MAE()
	case huber:
		return Huber()
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

// MAE is the Mean Absolute Error (L1 loss). Its gradient has the same
// magnitude no matter how large the error, so single wild outliers pull on
// the network far less than under MSE -- at the price of a constant-size
// gradient that never softens as predictions get close. The scaling of F
// and FPrime mirrors MSE's convention (sum over elements divided by rows,
// derivative divided by rows*columns).
func MAE() Loss {
	return Loss{
		Name: mae,
		F: func(y, yHat matrix.Matrix) float64 {
			return y.SubtractMatrix(yHat).
				Map(func(val float64, x, c int) float64 {
					return math.Abs(val)
				}).
				Sum() / float64(y.Rows)
		},
		FPrime: func(y, yHat matrix.Matrix) matrix.Matrix {
			// Like MSE().FPrime, this returns the *negative* gradient with
			// respect to the prediction (the direction outputDelta adds to
			// descend): sign(target - prediction), flat at exactly 0 where
			// L1 has no derivative.
			return yHat.SubtractMatrix(y).
				Map(func(val float64, x, c int) float64 {
					if val > 0 {
						return 1
					} else if val < 0 {
						return -1
					}
					return 0
				}).
				Divide(float64(y.Rows * y.Columns))
		},
	}
}

// Huber returns the Huber loss with the conventional delta of 1: quadratic
// like MSE inside |error| <= delta, linear like MAE outside it, so it keeps
// MSE's smooth convergence near the target without letting outliers
// dominate. Use HuberWithDelta to pick the crossover point.
func Huber() Loss {
	return HuberWithDelta(1)
}

// HuberWithDelta is Huber with a caller-chosen quadratic/linear crossover.
// delta must be positive. Note that Save/Load round-trips losses by name
// only, so a loaded network falls back to the default delta of 1.
func HuberWithDelta(delta float64) Loss {
	if delta <= 0 {
		panic("goneural: Huber delta must be positive")
	}

	return Loss{
		Name: huber,
		F: func(y, yHat matrix.Matrix) float64 {
			return y.SubtractMatrix(yHat).
				Map(func(val float64, x, c int) float64 {
					if math.Abs(val) <= delta {
						return 0.5 * val * val
					}
					return delta * (math.Abs(val) - 0.5*delta)
				}).
				Sum() / float64(y.Rows)
		},
		FPrime: func(y, yHat matrix.Matrix) matrix.Matrix {
			// Negative gradient w.r.t. the prediction, matching MSE's
			// descent-direction convention: the residual inside the
			// quadratic zone, clamped to +-delta outside it -- the branches
			// meet continuously at |residual| = delta.
			return yHat.SubtractMatrix(y).
				Map(func(val float64, x, c int) float64 {
					if math.Abs(val) <= delta {
						return val
					}
					if val > 0 {
						return delta
					}
					return -delta
				}).
				Divide(float64(y.Rows * y.Columns))
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
