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
     * @param learning_rate  the initial learning rate for the optimizer, typically a small positive value (e.g., 0.1 or 0.01)
     * @param decay_rate     the rate at which the learning rate decays, controlling how quickly it decreases over time
     * @param clip_value     the maximum value for gradient clipping; gradients larger than this value will be scaled down
     * @param batch_size     the size of each batch for training; determines how many samples are used per training step
     * @param momentum      a value between 0 and 1 (e.g., 0.9); that speeds up convergence and reduces oscillations
     *                      by incorporating a fraction of the previous weight . Momentum = 0 means no usage of momentum
     * @param weight_decay
     */
    public Hyperparameters(int epochs, float learning_rate, float decay_rate, float clip_value, int batch_size, float momentum, float weight_decay) {
        this.epochs = epochs;
        this.learningRate = learning_rate;
        this.clipValue = clip_value;
        this.batchSize = batch_size;
        this.decayRate = decay_rate;
        this.momentum = momentum;
        this.weightDecay = weight_decay;
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


    public void setWeightDecay(float weight_decay) {
        this.weightDecay = weight_decay;
    }
}
