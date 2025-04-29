package net.echo.brain4j.transformers.attention;

import net.echo.brain4j.initialization.WeightInitializer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.transformers.head.AttentionHead;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MultiHeadAttention {

    protected final List<AttentionHead> heads;
    protected final int headCount;
    protected final int embeddingDim;
    protected final int headDimension;

    public MultiHeadAttention(int headCount, int embeddingDim) {
        this.headCount = headCount;
        this.embeddingDim = embeddingDim;

        if (embeddingDim % headCount != 0) {
            throw new IllegalArgumentException("Embedding dimension must be divisible by head count! (%s %% %s = %s)"
                    .formatted(embeddingDim, headCount, embeddingDim % headCount));
        }

        this.headDimension = embeddingDim / headCount;
        this.heads = new ArrayList<>();

        initializeHeads();
    }

    public void compile(Random generator, WeightInitializer initializer) {
        for (AttentionHead head : heads) {
            head.compile(generator, initializer);
        }
    }

    public AttentionHead createAttentionHead() {
        return new AttentionHead(embeddingDim, headDimension);
    }

    public Tensor attend(StatesCache cache, Tensor input) {
        Tensor[] outputs = new Tensor[heads.size()];

        for (int i = 0; i < heads.size(); i++) {
            AttentionHead head = heads.get(i);
            outputs[i] = head.attend(cache, input);
        }

        return Tensors.concat(List.of(outputs));
    }

    public int getTotalNeurons() {
        return heads.stream()
                .mapToInt(AttentionHead::size)
                .sum();
    }

    public void setUseCache(boolean useCache) {
        for (AttentionHead head : heads) {
            head.setUseCache(useCache);
        }
    }

    protected void initializeHeads() {
        for (int i = 0; i < headCount; i++) {
            heads.add(createAttentionHead());
        }
    }

    public List<AttentionHead> getHeads() {
        return heads;
    }
}
