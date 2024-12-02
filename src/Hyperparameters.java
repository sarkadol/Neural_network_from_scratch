package src;

public class Hyperparameters {
    private int epochs;
    private float learningRate;
    private float clipValue;
    private int batchSize;
    private float decayRate;
    private float momentum; //recommended default 0.9
    private float weightDecay; //recommended default 0.0005


    /**
     * Constructs a Hyperparameters object with the specified values for epochs, learning rate, decay rate,
     * gradient clipping value, and batch size.
     *
     * @param epochs        the total number of training epochs (iterations over the entire dataset)
     * @param learningRate  the initial learning rate for the optimizer, typically a small positive value (e.g., 0.1 or 0.01)
     * @param decayRate     the rate at which the learning rate decays, controlling how quickly it decreases over time
     * @param clipValue     the maximum value for gradient clipping; gradients larger than this value will be scaled down
     * @param batchSize     the size of each batch for training; determines how many samples are used per training step
     * @param momentum
     * @param weightDecayRate
     */
    public Hyperparameters(int epochs, float learningRate, float decayRate, float clipValue, int batchSize, float momentum, float weightDecayRate) {
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.clipValue = clipValue;
        this.batchSize = batchSize;
        this.decayRate = decayRate;
        this.momentum = momentum;
        this.weightDecay = weightDecayRate;
    }

    public int getEpochs() {
        return epochs;
    }
    public float getLearningRate() {
        return learningRate;
    }
    public float getClipValue() {
        return clipValue;
    }
    public int getBatchSize() {
        return batchSize;
    }
    public float getDecayRate() {return decayRate;}
    public float getMomentum() {
        return momentum;
    }


    public float getWeightDecay() {
        return weightDecay;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public void setClipValue(float clipValue) {
        this.clipValue = clipValue;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public void setDecayRate(float decayRate) {
        this.decayRate = decayRate;
    }


    public void setMomentum(float momentum) {
        this.momentum = momentum;
    }


    public void setWeightDecay(float weightDecay) {
        this.weightDecay = weightDecay;
    }
}
