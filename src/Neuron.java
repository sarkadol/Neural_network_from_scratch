package src;

import java.util.Arrays;

public class Neuron {
    public Neuron() {

    }
    float bias;
    float[] weights; //array of weights w1...w_y
    float[] x;//array of input values from previous layer x1...x_y
    float[] y; //ted atribut, příště metoda

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
                "Bias: " + bias + ", " +
                "Weights count: " + (weights != null ? weights.length : 0) + ", " +
                "Inputs count: " + (x != null ? x.length : 0) + ", " +
                "Outputs count: " + (y != null ? y.length : 0));
    }
}
