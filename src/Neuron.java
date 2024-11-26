package src;

import java.util.Arrays;

public class Neuron {
    float bias;
    float[] weights; //array of weights w1...w_y
    float[] x;//array of input values from previous layer x1...x_y
    float inner_potential = 0;
    float[] y;

    public Neuron() {
        this.bias = 0;
        this.weights = null;
        this.x = null;
        this.y = null;
    }

    //getters and setters
    public float[] getWeights() {
        return weights;
    }
    public float getBias() {
        return bias;
    }
    public float[] getX() {
        return x;
    }

    public float getInnerPotential() {
        return inner_potential;
    }

    public float[] getY() {
        return y;
    }

    public void setBias(float bias) {
        this.bias = bias;
    }
    public void setWeights(float[] weights) {
        this.weights = weights;
    }
    public void setX(float[] x) {
        this.x = x;
    }
    public void setY(float[] y) {
        this.y = y;
    }

    /**
     * Computes the neuron's inner potential from the given input and weights, then it stores the inner potential
     * for later use
     * Formula: ξ_j = ∑ w_ji . y_i
     * @return the computed inner potential
     */
    public float computeInnerPotential() {
        float innerPotential = bias;
        innerPotential += Util.scalarProduct(weights, x);
        this.inner_potential = innerPotential;  // Update this neuron's respective attribute
        return innerPotential;
    }

    /**
     * Prints the neuron's bias, weight values, input values, and output values
     * in a multi-line format.
     */
    public void printInfo(){
        System.out.println("Neuron's properties:");
        System.out.println("Bias: " + bias);
        System.out.println("Weights: "+ Arrays.toString(weights));
        System.out.println("X: "+ Arrays.toString(x));
        System.out.println("Y: "+ Arrays.toString(y));
    }
    /**
     * SINGLE LINE.
     * Prints the bias, the count of weights, the count of inputs, and the count of outputs.
     * This provides a quick overview without displaying all values in detail.
     */
    public void printInfoLine() {
        System.out.println(
                "Neuron "+
                "Bias: " + bias + ", " +
                "Weights count: " + (weights != null ? weights.length : 0) + ", " +
                "Inputs count: " + (x != null ? x.length : 0) + ", " +
                "Outputs count: " + (y != null ? y.length : 0));
    }
}
