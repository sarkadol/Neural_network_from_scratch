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

    public void PrintProperties(){
        System.out.println("Neuron's properties:");
        System.out.println("Bias: " + bias);
        System.out.println("Weights: "+ Arrays.toString(weights));
        System.out.println("X: "+ Arrays.toString(x));
        System.out.println("Y: "+ Arrays.toString(y));
    }
}
