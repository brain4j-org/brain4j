package net.echo.brain4j.transformers.masked;

import com.google.common.base.Preconditions;
import net.echo.brain4j.transformers.attention.AttentionHead;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.transformers.attention.MultiHeadAttention;
import net.echo.math4j.math.vector.Vector;

import java.util.ArrayList;
import java.util.List;

public class MaskedMultiHeadAttention extends MultiHeadAttention {

    public MaskedMultiHeadAttention(WeightInitializer weightInit, int headCount, int modelDimension, double temperature) {
        super(weightInit, headCount, modelDimension, temperature);

        Preconditions.checkState(modelDimension % headCount == 0, "Model dimension must be divisible by head count!");

        initializeHeads();
        initializeOutProjectionWeights();
    }

    @Override
    public AttentionHead createAttentionHead() {
        return new MaskedAttentionHead(weightInit, modelDimension, headDimension, temperature);
    }

    @Override
    public int getTotalNeurons() {
        int total = outProjectionTensor.elements();

        for (AttentionHead head : heads) {
            total += head.size();
        }

        return total;
    }
}
