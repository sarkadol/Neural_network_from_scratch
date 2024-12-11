package src;

import src.helpers.Helper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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

    /**
     * Predicts all images - vectors representing images to predict
     * @param vectors List of float arrays
     * @return an array of ints - labels
     */
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
     * Computes a vector that is then fed into the softmax function to produce a part in the computation
     * of the VLayer output gradients
     * @return the vector that is then fed into the softmax
     */
    private float[] computeOutputLayerSoftmaxVector() {
        Layer VLayer = layers[layers.length - 2];
        Layer outputLayer = layers[layers.length - 1];
        int neuronsNumberV = VLayer.neurons.length;
        int neuronsNumberY = outputLayer.neurons.length;
        float[] resultVector = new float[neuronsNumberY];
        for (int j = 0; j < neuronsNumberY; j++) {
            resultVector[j] = 0;
            for (int l = 0; l < neuronsNumberV; l++) {
                resultVector[j] += outputLayer.neurons[j].weights[l] * VLayer.y[l];
            }
        }
        return resultVector;
    }


    public float[] computeVLayerOutputGradients(float[] targets) {
        Layer VLayer = layers[layers.length - 2];
        Layer outputLayer = layers[layers.length - 1];
        int neuronsNumberV = VLayer.neurons.length;
        int neuronsNumberY = outputLayer.neurons.length;
        float[] gradients = new float[neuronsNumberV];
        float[] outputLayerSoftmax = Util.softmax(computeOutputLayerSoftmaxVector());   // σ(v→)
        for (int l = 0; l < neuronsNumberV; l++) {

            float innerSum = 0;     //Σ_j∈Y(w_jl · σ(v→)_j) it is used later, but does not depend on any other index than l
            for (int j = 0; j < neuronsNumberY; j++) {
                innerSum += outputLayer.neurons[j].weights[l] * outputLayerSoftmax[j];
            }

            gradients[l] = 0;
            for (int i = 0; i < neuronsNumberY; i++) {

                gradients[l] += targets[i] * (outputLayer.neurons[i].weights[l] - innerSum);    // di · (wil − innerSum); innerSum is described in the above comment
            }
            gradients[l] *= -1;
        }
        return gradients;
    }


    /**
     * Clips the gradients to prevent exploding.
     * −clipValue ≤ gradients[i][j] ≤ clipValue
     * (see the slide 108 "Issues in gradient descent – too fast descent")
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
     * BACKPROPAGATION and WEIGHTS UPDATE - without batches, a prototype used earlier
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
        outputLayer.updateWeights(output_layer_weight_gradients, hyperparameters.getLearningRate(), hyperparameters.getMomentum(), hyperparameters.getWeightDecay());

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
            previousLayer.updateWeights(previous_weight_gradients, hyperparameters.getLearningRate(), hyperparameters.getMomentum(), hyperparameters.getWeightDecay());

            current_output_gradient = previous_output_gradient; //move to another layer
        }
    }

    /**
     * Computes weight gradients for all layers during backpropagation.
     *
     * @param target Target probabilities.
     * @param outputs Predicted probabilities from forward pass.
     * @param hyperparameters Hyperparameters controlling training.
     * @return List of weight gradients for all layers.
     */
    public List<float[][]> computeWeightGradients(float[] target, float[] outputs, Hyperparameters hyperparameters) {

        //float loss = Util.crossEntropy(target, outputs);
        //System.out.println("cross entropy: "+loss);

        int numberOfLayers = layers.length;
        List<float[][]> weightGradients = new ArrayList<>(numberOfLayers - 1);

        // ---------------OUTPUT LAYER -------------------
        Layer outputLayer = layers[numberOfLayers - 1];

        // 1) output gradients (actually inner potential gradients here)
        float[] output_layer_gradients = computeOutputLayerGradients(outputs,target); //gradient wrt ξ, maybe also wrt. y, we do not need wrt. y to compute weight gradient here

        // 2) weight gradients
        float[][] output_layer_weight_gradients = outputLayer.computeOutputLayerWeightGradients(output_layer_gradients); //gradient wrt w
        // Clip gradients for output layer
        output_layer_weight_gradients = clipGradients(output_layer_weight_gradients, hyperparameters.getClipValue()); // Example clip value

        weightGradients.add(output_layer_weight_gradients);
        float[] VLayerOutputGradients = null;
        if (hyperparameters.isGradients_new_method()) {
            // process also the layer immediately below the output layer - VLayer
            // -----------------UPPERMOST HIDDEN LAYER-------------
            // 1) output gradients
            VLayerOutputGradients = computeVLayerOutputGradients(target);
            // 2) weight gradients
            float[][] VLayerWeightGradients = layers[numberOfLayers - 2].computeWeightGradients(VLayerOutputGradients);
            weightGradients.add(VLayerWeightGradients);
        }


        // -----------------HIDDEN LAYERS----------------------
        float[] currentOutputGradients;
        if (hyperparameters.isGradients_new_method()) {
            currentOutputGradients = VLayerOutputGradients; // TODO PLACEHOLDER
        } else {
            currentOutputGradients = output_layer_gradients; //move from output layer
        }
        for (int i = numberOfLayers - (hyperparameters.isGradients_new_method() ? 2 : 1); i > 1; i--) {   // if we use
            // the new method for computing the gradients, we start one layer lower
            Layer currentLayer = layers[i];
            Layer previousLayer = layers[i - 1];

            // 1) output gradients
            float[] previous_output_gradient = backpropagateHiddenLayer(currentOutputGradients, currentLayer, previousLayer);

            // 2) weight gradients
            float[][] previous_weight_gradients = previousLayer.computeWeightGradients(previous_output_gradient);
            // Clip gradients for the current hidden layer
            previous_weight_gradients = clipGradients(previous_weight_gradients, hyperparameters.getClipValue()); // Example clip value

            weightGradients.add(previous_weight_gradients);

            currentOutputGradients = previous_output_gradient; //move to another layer
        }
        Collections.reverse(weightGradients);
        return weightGradients;
    }
    /**
     * Trains the network using mini-batches of data.
     *
     * @param trainVectors List of input vectors for training.
     * @param trainLabels List of corresponding labels for training.
     * @param hp Hyperparameters controlling training.
     * @param verbose If true, prints debug information during training.
     * @return Average loss over the mini-batches.
     */
    public float trainBatch(List<float[]> trainVectors, List<Integer> trainLabels,
                           Hyperparameters hp,
                           boolean verbose) {
        List<float[][]> weightGradientsPerLayer = new ArrayList<>(layers.length - 1);
        for (int i = 0; i < layers.length - 1; i++) {
            weightGradientsPerLayer.add(layers[i + 1].initializeWeightGradients());
        }
        float totalLoss = 0;
        for (int i = 0; i < trainVectors.size(); i++) {
            float[] inputs = trainVectors.get(i);
            float[] target = Util.labelToVector(trainLabels.get(i));
            float[] outputs = forwardPass(inputs);

            List<float[][]> currentWeightGradientsPerLayer = computeWeightGradients(target, outputs, hp);
            for (int j = 0; j < layers.length - 1; j++) {
                weightGradientsPerLayer.set(
                        j,
                        layers[j + 1].addWeightGradients(
                                currentWeightGradientsPerLayer.get(j),
                                weightGradientsPerLayer.get(j)));
            }

            totalLoss += Util.crossEntropy(target, outputs);

            if (verbose) {
                System.out.println("\n--- Debug Info ---");
                System.out.println("Image: " + i);
                System.out.println("Target: " + Arrays.toString(target));
                System.out.println("Outputs: " + Arrays.toString(outputs));
                System.out.println("Cross-entropy loss: " + Util.crossEntropy(target, outputs));
                System.out.println("Learning rate: " + hp.getLearningRate());
            }
        }

        for (int i = 0; i < layers.length - 1; i++) {
            layers[i + 1].updateWeights(
                    weightGradientsPerLayer.get(i),
                    hp.getLearningRate(),
                    hp.getMomentum(),
                    hp.getWeightDecay()
                    );
        }
        return totalLoss / trainVectors.size();
    }

    /**
     * Trains the network over multiple epochs using the entire training dataset.
     *
     * @param trainVectors List of input vectors for training.
     * @param trainLabels List of corresponding labels for training.
     * @param hp Hyperparameters controlling training.
     * @param verbose If true, prints debug information during training.
     */
    public void trainNetwork(List<float[]> trainVectors, List<Integer> trainLabels,
                             Hyperparameters hp,   // Hyperparameters
                             boolean verbose) {
        System.out.println("\nTraining "+hp.getEpochs()+" epochs...");

        final long MAX_EPOCH_TIME = 20 * 60 * 1000; // Time after which it cancells the process in milliseconds

        Dataset dataset = new Dataset(trainVectors, trainLabels);
        int batchSize = hp.getBatchSize();
        int epochs = hp.getEpochs();
        float[] losses = new float[epochs];
        float[] learning_rates = new float[epochs];

        long trainingStartTime = System.currentTimeMillis(); // Record the start time

        for (int epoch = 0; epoch < epochs; epoch++) {
            dataset.shuffle();
            List<List<float[]>> vectorBatches = Util.divideToBatches(dataset.getVectors(), batchSize);
            List<List<Integer>> labelBatches = Util.divideToBatches(dataset.getLabels(), batchSize);

            float totalLoss = 0;
            int numberOfBatches = vectorBatches.size();
            if (vectorBatches.size() != labelBatches.size()) {
                throw new RuntimeException("Number of batches does not match number of labels");
            }
            for (int i = 0; i < numberOfBatches; i++) {
                totalLoss += trainBatch(vectorBatches.get(i), labelBatches.get(i), hp, verbose);

                if ((System.currentTimeMillis()-trainingStartTime) > MAX_EPOCH_TIME) {
                    System.out.println("\nEpoch " + epoch + " exceeded 10 minutes in total. Training stopped, returning current model state.");
                    return;
                }
            }

            if(hp.useLearningDecayRate()){//if we want to use the exponential learning rate
                hp.setLearningRate(hp.getLearningRate() * (float)Math.pow(0.1, epoch / hp.getLearningDecayRate())); //slightly decrease the learning rate - exponential scheduling
                //ϵ(t) = ϵ0 · 0.1^(t/s) -> slide 118 from NEW_continuously_updated_slides.pdf
            }
            losses[epoch] = totalLoss / numberOfBatches;    // The last batch may be smaller, so it is not the same
                                                            // as dividing the total overall loss by the number
                                                            // of samples, but it does not matter
            learning_rates[epoch] = hp.getLearningRate();
            // Calculate and print the time elapsed since the training started
            long elapsedMillis = System.currentTimeMillis() - trainingStartTime;
            long minutes = (elapsedMillis / 1000) / 60;
            long seconds = (elapsedMillis / 1000) % 60;

            System.out.println(String.format("Epoch %-3d completed in %02d:%02d.", epoch, minutes, seconds));
        }
        System.out.println("\nTraining completed\n");

        //System.out.println("losses = " + Arrays.toString(losses));
        //System.out.println("learning_rates = " + Arrays.toString(learning_rates));

        //pass this to file which is read by Python to plot the changes during epochs
        Helper.writeToFileForPython(losses,learning_rates,trainVectors.size(),Arrays.toString(getLayersLength()),hp);
    }


    /**
     * Initializes weights and biases for all layers in the network, excluding the input layer.
     */
    public void initializeWeights(){
        for (int i = 1; i < layers.length; i++){//skip 0th because it has no weights nor bias
            layers[i].InitializeWeights();
        }
        System.out.println("Weights initialized");
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
