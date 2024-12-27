package src;

import java.util.Arrays;

/**
 * Represents a single neuron in a neural network.
 *
 * <p>A neuron contains weights, a bias, and input values, which are used to calculate its output.
 * It also supports operations like computing inner potential, printing its state, and maintaining
 * momentum for weight updates.</p>
 */
public class Neuron {
    float bias;
    float[] weights; //array of weights w1...w_y
    float[] x;//array of input values from previous layer x1...x_y

    public Neuron() {
        this.bias = 0;
        this.weights = null;
        this.x = null;
    }

    public void setBias(float bias) {
        this.bias = bias;
    }
    public void setWeights(float[] weights) {
        this.weights = weights;
    }

}
