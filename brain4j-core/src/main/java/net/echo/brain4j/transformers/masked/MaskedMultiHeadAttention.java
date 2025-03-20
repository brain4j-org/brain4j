package net.echo.brain4j.transformers.masked;

import com.google.common.base.Preconditions;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.transformers.attention.MultiHeadAttention;
import net.echo.math4j.math.vector.Vector;

import java.util.ArrayList;
import java.util.List;

public class MaskedMultiHeadAttention extends MultiHeadAttention {

    private final List<MaskedAttentionHead> heads;

    public MaskedMultiHeadAttention(WeightInitializer weightInit, int headCount, int modelDimension, double temperature) {
        super(weightInit, headCount, modelDimension, temperature);
        this.heads = new ArrayList<>();

        Preconditions.checkState(modelDimension % headCount == 0, "Model dimension must be divisible by head count!");

        initializeHeads();
        initializeOutProjectionWeights();
    }

    @Override
    public List<Vector> attend(List<Vector> inputs) {
        List<List<Vector>> headOutputs = new ArrayList<>();

        for (MaskedAttentionHead head : heads) {
            headOutputs.add(head.attend(inputs));
        }

        return concatenate(headOutputs, inputs);
    }
    
    @Override
    public List<Tensor> attendTensors(List<Tensor> inputs) {
        List<List<Tensor>> headOutputs = new ArrayList<>();

        for (MaskedAttentionHead head : heads) {
            headOutputs.add(head.attendTensors(inputs));
        }

        return concatenateTensors(headOutputs, inputs);
    }

    @Override
    public int getTotalNeurons() {
        int total = outProjectionWeights.length * modelDimension;

        for (MaskedAttentionHead head : heads) {
            total += head.size();
        }

        return total;
    }

    @Override
    protected void initializeHeads() {
        for (int i = 0; i < headCount; i++) {
            heads.add(new MaskedAttentionHead(weightInit, modelDimension, headDimension, temperature));
        }
    }
}
