package goneural

import "math"

// InitXavier re-initializes the network's weights with Glorot/Xavier
// uniform initialization (Glorot & Bengio, 2010): each weight layer is
// drawn from U(-limit, limit) with limit = sqrt(6 / (fanIn + fanOut)), and
// the biases are zeroed. Scaling by the layer's fan keeps activation and
// gradient variance roughly constant from layer to layer, which is tuned
// for symmetric squashing activations like sigmoid and tanh -- New's plain
// U(-1, 1) tends to saturate wide layers from the very first epoch.
func (n *NeuralNetwork) InitXavier() {
	for i := range n.Weights {
		fanIn := float64(n.Layers[i].Nodes)
		fanOut := float64(n.Layers[i+1].Nodes)
		limit := math.Sqrt(6 / (fanIn + fanOut))

		n.Weights[i].Randomize(-limit, limit)
		n.Biases[i].Zero()
	}
}

// InitHe re-initializes the network's weights with He/Kaiming uniform
// initialization (He et al., 2015): U(-limit, limit) with
// limit = sqrt(6 / fanIn), biases zeroed. The larger variance compensates
// for ReLU-family activations zeroing (or nearly zeroing) half their
// inputs, so it is the usual choice for ReLU, LeakyReLU and ELU layers.
func (n *NeuralNetwork) InitHe() {
	for i := range n.Weights {
		fanIn := float64(n.Layers[i].Nodes)
		limit := math.Sqrt(6 / fanIn)

		n.Weights[i].Randomize(-limit, limit)
		n.Biases[i].Zero()
	}
}
