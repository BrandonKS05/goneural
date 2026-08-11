package goneural

import "math"

// Activation is an activation function
// it contains the normal f(x) and the derivative f'(x)
type Activation struct {
	Name   activationName
	F      func(x float64) float64
	FPrime func(x float64) float64
}

type activationName string

const (
	sigmoid   activationName = "sigmoid"
	relu      activationName = "relu"
	softmax   activationName = "softmax"
	id        activationName = "id"
	tanh      activationName = "tanh"
	leakyRelu activationName = "leaky_relu"
	elu       activationName = "elu"
	softplus  activationName = "softplus"
)

func getActivationFromName(a activationName) Activation {
	switch a {
	case sigmoid:
		return Sigmoid()
	case relu:
		return ReLU()
	case softmax:
		return Softmax()
	case id:
		return Identity()
	case tanh:
		return Tanh()
	case leakyRelu:
		return LeakyReLU()
	case elu:
		return ELU()
	case softplus:
		return Softplus()
	default:
		return Identity()
	}
}

// Sigmoid is a sigmoid activation functio
func Sigmoid() Activation {
	return Activation{
		Name: sigmoid,
		F: func(x float64) float64 {
			return 1 / (1 + math.Exp(-x))
		},
		FPrime: func(x float64) float64 {
			return x * (1 - x)
		},
	}
}

// ReLU is a ReLU activation function
func ReLU() Activation {
	return Activation{
		Name: relu,
		F: func(x float64) float64 {
			return math.Max(0, x)
		},
		FPrime: func(x float64) float64 {
			if x > 0 {
				return 1
			}
			return 0
		},
	}
}

// Identity is the identity (linear) function
// f(x) = x
func Identity() Activation {
	return Activation{
		Name: id,
		F: func(x float64) float64 {
			return x
		},
		FPrime: func(_ float64) float64 {
			return 1
		},
	}
}

// Tanh is the hyperbolic tangent activation function. It squashes inputs
// into (-1, 1) much like sigmoid squashes into (0, 1), but is zero-centered,
// which tends to keep hidden-layer gradients better behaved. FPrime, per
// this package's convention, receives the activation's output y = tanh(x);
// the derivative in those terms is 1 - y^2.
func Tanh() Activation {
	return Activation{
		Name: tanh,
		F:    math.Tanh,
		FPrime: func(y float64) float64 {
			return 1 - y*y
		},
	}
}

// LeakyReLU is a ReLU that lets a small gradient through for negative
// inputs (f(x) = 0.01*x when x < 0) instead of zeroing them, which keeps
// units from going permanently "dead" when they drift negative. Use
// LeakyReLUWithAlpha to pick a different negative-side slope.
func LeakyReLU() Activation {
	return LeakyReLUWithAlpha(0.01)
}

// LeakyReLUWithAlpha is LeakyReLU with a caller-chosen negative-side slope.
// alpha must be positive: the FPrime convention hands over the activation's
// output, and only a sign-preserving slope lets the negative branch be
// recovered from the output alone. Note that Save/Load round-trips
// activations by name only, so a loaded network falls back to the default
// 0.01 slope.
func LeakyReLUWithAlpha(alpha float64) Activation {
	if alpha <= 0 {
		panic("goneural: LeakyReLU alpha must be positive")
	}

	return Activation{
		Name: leakyRelu,
		F: func(x float64) float64 {
			if x > 0 {
				return x
			}
			return alpha * x
		},
		FPrime: func(y float64) float64 {
			if y > 0 {
				return 1
			}
			return alpha
		},
	}
}

// ELU is the Exponential Linear Unit (Clevert et al., 2015): identity for
// positive inputs, alpha*(e^x - 1) for negative ones. Unlike ReLU it stays
// smooth through zero and saturates to -alpha instead of dying, which pushes
// mean activations toward zero. Uses the conventional alpha of 1; see
// ELUWithAlpha to pick another.
func ELU() Activation {
	return ELUWithAlpha(1)
}

// ELUWithAlpha is ELU with a caller-chosen negative-side scale. alpha must
// be positive so the negative branch's derivative can be recovered from the
// activation's output (y = alpha*(e^x - 1) gives dF/dx = y + alpha). Note
// that Save/Load round-trips activations by name only, so a loaded network
// falls back to the default alpha of 1.
func ELUWithAlpha(alpha float64) Activation {
	if alpha <= 0 {
		panic("goneural: ELU alpha must be positive")
	}

	return Activation{
		Name: elu,
		F: func(x float64) float64 {
			if x > 0 {
				return x
			}
			return alpha * (math.Exp(x) - 1)
		},
		FPrime: func(y float64) float64 {
			if y > 0 {
				return 1
			}
			return y + alpha
		},
	}
}

// Softplus is a smooth approximation of ReLU: f(x) = ln(1 + e^x). It is
// always positive and differentiable everywhere, with derivative
// sigmoid(x), which in terms of the output y works out to 1 - e^(-y).
func Softplus() Activation {
	return Activation{
		Name: softplus,
		F: func(x float64) float64 {
			// max(x, 0) + log1p(e^(-|x|)) is the overflow-safe rewrite of
			// ln(1 + e^x): the exponential's argument is never positive.
			return math.Max(x, 0) + math.Log1p(math.Exp(-math.Abs(x)))
		},
		FPrime: func(y float64) float64 {
			return 1 - math.Exp(-y)
		},
	}
}

// Softmax is a softmax activation function. Unlike the others it isn't
// elementwise -- every output depends on all the other values in the same
// layer, since they all share one normalizing sum -- so F/FPrime here just
// panic; NeuralNetwork.predict and the optimizers special-case Layers with
// Name == softmax instead of going through the generic per-element path.
// It should only be used on the output layer, paired with CrossEntropy.
func Softmax() Activation {
	return Activation{
		Name: softmax,
		F: func(x float64) float64 {
			panic("goneural: softmax can't be applied elementwise; use it only as the output layer's activation")
		},
		FPrime: func(x float64) float64 {
			panic("goneural: softmax can't be applied elementwise; use it only as the output layer's activation")
		},
	}
}
