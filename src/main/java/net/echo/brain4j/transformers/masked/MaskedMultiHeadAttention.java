package net.echo.brain4j.transformers.masked;

import com.google.common.base.Preconditions;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.transformers.attention.MultiHeadAttention;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class MaskedMultiHeadAttention extends MultiHeadAttention {

    private final List<MaskedAttentionHead> heads;

    public MaskedMultiHeadAttention(WeightInit weightInit, int headCount, int modelDimension, double temperature) {
        super(weightInit, headCount, modelDimension, temperature);
        this.heads = new ArrayList<>();

        Preconditions.checkState(modelDimension % headCount == 0, "Model dimension must be divisible by head count!");

        initializeHeads();
        initializeOutProjectionWeights();
    }

    public List<Vector> attend(List<Vector> inputs) {
        List<List<Vector>> headOutputs = new ArrayList<>();

        for (MaskedAttentionHead head : heads) {
            headOutputs.add(head.attend(inputs));
        }

        return concatenate(headOutputs, inputs);
    }

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
