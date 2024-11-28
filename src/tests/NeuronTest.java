package src.tests;

import src.Neuron;

public class NeuronTest {
    public static void main(String[] args) {
        Neuron neuron = new Neuron();
        neuron.setWeights(new float[10]);
        neuron.setX(new float[10]);
        //neuron.setY(new float[10]);
        neuron.setBias(0);
        neuron.printInfo();
    }

}
