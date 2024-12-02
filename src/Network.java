package src;

import java.util.Arrays;
import java.util.List;


import static src.Util.activationDerivative;

public class Network {
    public Layer[] layers;

    /**
     * Creates a network with layers
     * @param layers Layers of the network
     */
    public Network(Layer[] layers) {
        this.layers = layers;
        this.initializeWeights();
    }

    /**
     * Performs forward propagation
     * Calculates the output of the network by sequentially passing data through each layer.
     *
     * @param inputs float array representing the input features for a single image
     * @return array representing the network's output (class probabilities)
     */
    public float[] forwardPass(float[] inputs){
        //System.out.println("Forward pass proceeding...");
        for (int i = 1; i < layers.length; i++){    // V tuto chvíli je nepotřebná vstupní vrstva
            layers[i].setX(inputs); // Save respective attributes
            inputs = layers[i].computeOutput(inputs);
        }
        //System.out.println("Forward pass complete.");
        return inputs;
    }

    /**
     * Trained network returns the most probable label of an image.
     * @param inputs one image
     * @return the most probable label of an image
     */
    public int predict(float[] inputs){
        //Fashion MNIST has 10 labels 0-9:
        //"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        //"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"

        if(inputs.length != layers[0].getOutputLength()){
            throw new IllegalArgumentException("Input length does not match first layer's length.");
        }
        //System.out.println("\nPredicting...");
        float[] outputs = forwardPass(inputs);
        //System.out.println("Outputs: "+Arrays.toString(outputs));
        int label = 0;
        for (int i = 1; i < outputs.length; i++) {
            if (outputs[i] > outputs[label]) { //if values are the same, it predicts the first one
                label = i; // Update if the current output is larger
            }
        }
        //System.out.println("Predicting complete.");
        return label;
    }

    public int[] predictAll(List<float[]> vectors){
        int[] labels = new int[vectors.size()];
        for(int i = 0; i < vectors.size(); i++){
            labels[i]=predict(vectors.get(i));
        }
        return labels;
    }

    /**
     * Computes gradients of the output layer. (How should the y of output layer change)
     * @param predicted array of predicted probabilities
     * @param target array of desired probabilities
     * @return array of gradients
     */
    private float[] computeOutputLayerGradients(float[] predicted, float[] target) {
        float[] gradients = new float[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            gradients[i] = predicted[i] - target[i];    // TODO check if it is truly true; it is the same as partial
                                                        // derivative wrt. outputs and we are computing partial
                                                        // derivative wrt. inner potentials
        }
        return gradients;
    }

    /**
     * Backpropagation of a hidden layer:
     * (∂E_k / ∂y_j) = Σ_(r ∈ j→) [(∂E_k / ∂y_r) * σ'_r(ξ_r) * w_rj]
     *
     * r ∈ j→ indicates that r is in the set of nodes connected to j. (arch from j to r)
     * σ'_r(ξ_r) is the derivative of the activation function with respect to ξ_r.
     * w_rj represents the weight connecting nodes r and j.
     *
     * @param currentGradients gradients of outputs for the current layer
     * @param currentLayer Layer object of current layer
     * @param previousLayer Layer just before the current layer
     * @return gradients of outputs for previous layer
     */
    private float[] backpropagateHiddenLayer(float[] currentGradients, Layer currentLayer, Layer previousLayer) {
        int currentNeuronCount = currentLayer.neurons.length;
        int previousNeuronCount = previousLayer.neurons.length;

        float[] previousGradients = new float[previousNeuronCount]; // Gradient for previous layer

        for (int j = 0; j < previousNeuronCount; j++) {
            previousGradients[j] = 0; // Initialize gradient
            for (int r = 0; r < currentNeuronCount; r++) {
                // Backpropagate gradient from current layer
                previousGradients[j] +=
                        currentGradients[r] *
                        currentLayer.neurons[r].weights[j] *
                        activationDerivative(currentLayer.neurons[r].getInnerPotential(), currentLayer.activation_function);
            }
        }
        return previousGradients;
    }

    /**
     * Clips the gradients to prevent exploding.
     * −clipValue ≤ gradients[i][j] ≤ clipValue
     * (see the slide "Issues in gradient descent – too fast descent")
     * @param gradients 2D array f gradients
     * @param clipValue treshold value
     * @return
     */
    public static float[][] clipGradients(float[][] gradients, float clipValue) {
        for (int i = 0; i < gradients.length; i++) {
            for (int j = 0; j < gradients[i].length; j++) {
                gradients[i][j] = Math.max(-clipValue, Math.min(clipValue, gradients[i][j]));
            }
        }
        return gradients;
    }


    /**
     * BACKPROPAGATION and WEIGHTS UPDATE
     * First, it handles the output layer separately, and then it loops over the hidden layers.
     * At each layer:
     * 1) output gradients are computed,
     * 2) weight gradients are computed and
     * 3) weights are updated.
     * @param target list of desired probabilities given by label
     * @param outputs list of computed probabilities from forward pass
     * @param hyperparameters hyperparameters - learning rate and clip value used
     */
    public void train(float[] target, float[] outputs, Hyperparameters hyperparameters) {

        float loss = Util.crossEntropy(target, outputs);
        //System.out.println("cross entropy: "+loss);

        // ---------------OUTPUT LAYER -------------------
        Layer outputLayer = layers[layers.length - 1];

        // 1) output gradients
        float[] output_layer_gradients = computeOutputLayerGradients(outputs,target); //gradient wrt y

        // 2) weight gradients
        float[][] output_layer_weight_gradients = outputLayer.computeOutputLayerWeightGradients(output_layer_gradients); //gradient wrt w
        // Clip gradients for output layer
        output_layer_weight_gradients = clipGradients(output_layer_weight_gradients, hyperparameters.getClipValue()); // Example clip value
        //TODO how to choose a good clip value? recommended 1-5, but possible up to 20... - HYPERPARAMETER

        // 3) weight update
        outputLayer.updateWeights(output_layer_weight_gradients, hyperparameters.getLearningRate(), hyperparameters.getMomentum());

        // -----------------HIDDEN LAYERS----------------------

        float[] current_output_gradient = output_layer_gradients; //move from output layer

        for (int i = layers.length - 1; i > 1; i--) {
            Layer currentLayer = layers[i];
            Layer previousLayer = layers[i - 1];

            // 1) output gradients
            float[] previous_output_gradient = backpropagateHiddenLayer(current_output_gradient, currentLayer, previousLayer);

            // 2) weight gradients
            float[][] previous_weight_gradients = previousLayer.computeWeightGradients(previous_output_gradient);
            // Clip gradients for the current hidden layer
            previous_weight_gradients = clipGradients(previous_weight_gradients, 5.0f); // Example clip value

            // 3) weight update
            previousLayer.updateWeights(previous_weight_gradients, hyperparameters.getLearningRate(), hyperparameters.getMomentum());

            current_output_gradient = previous_output_gradient; //move to another layer
        }
    }


    public void trainBatch(List<float[]> trainVectors, List<Integer> trainLabels,
                           Hyperparameters hyperparameters,
                           boolean verbose) {

    }


    public void trainNetwork(List<float[]> trainVectors, List<Integer> trainLabels,
                             Hyperparameters hp,   // Hyperparameters
                             boolean verbose) {
        System.out.println("\nTraining...");

        int epochs = hp.getEpochs();
        float[] losses = new float[epochs];
        float[] learning_rates = new float[epochs];
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalLoss = 0;
            for (int i = 0; i < trainVectors.size(); i++) {
                float[] inputs = trainVectors.get(i);
                float[] target = Util.labelToVector(trainLabels.get(i));
                float[] outputs = forwardPass(inputs);
                train(target, outputs, hp);
                totalLoss += Util.crossEntropy(target, outputs);

                if (verbose) {
                    System.out.println("\n--- Debug Info ---");
                    System.out.println("Epoch: " + epoch + ", Image: " + i);
                    System.out.println("Target: " + Arrays.toString(target));
                    System.out.println("Outputs: " + Arrays.toString(outputs));
                    System.out.println("Cross-entropy loss: " + Util.crossEntropy(target, outputs));
                    System.out.println("Learning rate: " + hp.getLearningRate());
                }
            }
            if(false){//if we want to use the exponential learning rate - maybe to hyperparameter
                hp.setLearningRate(hp.getLearningRate() * (float)Math.pow(0.1, epoch / hp.getDecayRate())); //slightly decrease the learning rate - exponential scheduling
                //ϵ(t) = ϵ0 · 0.1^(t/s) -> slide 118 from NEW_continuously_updated_slides.pdf
            }

            System.out.println("Epoch " + epoch + ": Loss = " + (totalLoss / trainVectors.size()));
            learning_rates[epoch] = hp.getLearningRate();
            losses[epoch] = totalLoss / trainVectors.size();
        }
        System.out.println("\nTraining completed");

        //System.out.println("losses = " + Arrays.toString(losses));
        //System.out.println("learning_rates = " + Arrays.toString(learning_rates));

        //pass this to file which is read by Python to plot the changes during epochs
        DataLoader.writeToFileForPython(losses,learning_rates,trainVectors.size(),hp.getBatchSize(),hp.getDecayRate(),Arrays.toString(getLayersLength()));
    }


    /**
     * Initializes weights and biases for all layers in the network, excluding the input layer.
     */
    public void initializeWeights(){
        for (int i = 1; i < layers.length; i++){//skip 0th because it has no weights nor bias
            layers[i].InitializeWeights();
        }
        System.out.println("Initialization complete");
    }

    /**
     * Prints brief information about a network instance.
     */
    public void printInfo() {
        System.out.println("---Network - number of layers: " + layers.length);
        for (Layer layer : layers){
            layer.printInfoLine();
        }

    }

    /**
     * Iterates over each layer in a network and its length (number of neurons)
     * @return array of layers lengths, e. g. [784, 16, 8, 10]
     */
    public int[] getLayersLength(){
        int[] lengths = new int[layers.length];
        for(int i=0;i<layers.length;i++){
            lengths[i] = layers[i].getOutputLength();
        }
        return lengths;
    }
}
