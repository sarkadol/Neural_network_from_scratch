package src;

import java.util.Random;

public class Layer {
    public Neuron[] neurons;
    String activation_function;
    float[] y; //number of inputs
    float[][] weights; //i - number of y from the first layer, j - number of neurons second layer

    /**
     * creates a hidden/output layer
     * @param activation_function specifies activation function used in this layer
     * @param input_length length of the previous layer
     * @param neurons_number number of neurons in this layer
     */
    public Layer(int input_length, int neurons_number, String activation_function) {
        this.activation_function = activation_function;
        for(int i = 0; i < neurons_number; i++) {
            this.neurons[i] = new Neuron();
            //weight number for each neuron = input_length needed for initialization
        }
    }

    /**
     * Creates an input layer
     * @param input_neurons_number number of values e.g. from csv (needed for the next layer for weights)
     */
    public Layer(int input_neurons_number) {
        this.y = new float[input_neurons_number];
    }

    /**
     * Returns the length of a layer (number of neurons in this layer)
     * @return length of a layer
     */
    public int getLength(){
        return neurons.length;
    }
    /*
    public Neuron getNeuron(int index){
        return null;
    }*/

    /**
     * Randomly initializes weights for each neuron in this layer
     */
    public void InitializeWeights(){
        Random random = new Random(); //TODO check if it is correct
        for(int i = 0; i < neurons.length; i++){//each neuron i ...= for(Neuron neuron : neurons)
            for(int j = 0; j < y.length; j++){//each input j in particular neuron
                weights[i][j] = random.nextFloat();
                neurons[i].weights[j] = weights[i][j]; //save this weight into one Neuron
            }
        }

    }
}
