package net.echo.brain4j.transformers.attention;

import com.google.common.base.Preconditions;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.transformers.head.AttentionHead;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MultiHeadAttention {

    protected final List<AttentionHead> heads;
    protected final WeightInitializer weightInit;
    protected final double temperature;
    protected final int headCount;
    protected final int modelDimension;
    protected final int headDimension;

    protected Tensor outProjectionTensor;

    public MultiHeadAttention(WeightInitializer weightInit, int headCount, int modelDimension, double temperature) {
        this.weightInit = weightInit;
        this.headCount = headCount;
        this.modelDimension = modelDimension;
        this.temperature = temperature;

        Preconditions.checkState(modelDimension % headCount == 0, "Model dimension must be divisible by head count!");

        this.headDimension = modelDimension / headCount;
        this.heads = new ArrayList<>();
        this.outProjectionTensor = TensorFactory.matrix(headCount * headDimension, modelDimension);

        initializeHeads();
        initializeOutProjectionWeights();
    }

    public AttentionHead createAttentionHead() {
        return new AttentionHead(weightInit, modelDimension, headDimension, temperature);
    }

    public Tensor attend(Tensor input) {
        List<Tensor> headOutputs = new ArrayList<>();

        for (AttentionHead head : heads) {
            headOutputs.add(head.attend(input));
        }

        return TensorFactory.concat(headOutputs);
    }

    public int getTotalNeurons() {
        int total = 0;

        total += outProjectionTensor.elements();

        for (AttentionHead head : heads) {
            total += head.size();
        }

        return total;
    }

    protected void initializeHeads() {
        for (int i = 0; i < headCount; i++) {
            heads.add(createAttentionHead());
        }
    }

    protected void initializeOutProjectionWeights() {
        Random rng = new Random();
        double bound = weightInit.getBound(headCount * headDimension, modelDimension);

        for (int i = 0; i < headCount * headDimension; i++) {
            for (int j = 0; j < modelDimension; j++) {
                double value = (rng.nextDouble() * 2 * bound) - bound;
                outProjectionTensor.set(value, i, j);
            }
        }
    }
}
