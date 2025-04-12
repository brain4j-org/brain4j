package net.echo.brain4j.transformers.attention;

import com.google.common.base.Preconditions;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.transformers.head.AttentionHead;
import net.echo.math4j.BrainUtils;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MultiHeadAttention {

    protected final List<AttentionHead> heads;
    protected final int headCount;
    protected final int modelDimension;
    protected final int headDimension;

    public MultiHeadAttention(int headCount, int modelDimension) {
        this.headCount = headCount;
        this.modelDimension = modelDimension;

        Preconditions.checkState(modelDimension % headCount == 0, "Model dimension must be divisible by head count!");

        this.headDimension = modelDimension / headCount;
        this.heads = new ArrayList<>();

        initializeHeads();
    }

//    public Tensor propagate(StatesCache cache, Layer previous, Tensor delta) {
//        // Supponiamo che 'delta' abbia forma [batch, modelDimension]
//        // dove 'modelDimension' = headCount * headDimension.
//        // Per propagare il gradiente, lo dividiamo lungo l'ultima dimensione in 'headCount' parti.
//
//        // Split del delta in una lista di gradienti per ogni head.
//        List<Tensor> headDeltas = TensorFactory.split(delta, headCount);
//
//        // Inizializza il gradiente complessivo rispetto all'input
//        // Si assume che il TensorFactory possa creare un tensore di zeri con la stessa forma dell'input originario.
//        Tensor gradInput = TensorFactory.zerosLike(cache.getInputTensor(previous));
//
//        // Per ogni head, propaga il gradiente e accumula il gradiente rispetto all'input.
//        for (int i = 0; i < headCount; i++) {
//            AttentionHead head = heads.get(i);
//            Tensor headDelta = headDeltas.get(i);
//
//            // Assumiamo che ogni AttentionHead implementi un metodo propagate
//            // che ritorna il gradiente rispetto al suo input.
//            Tensor headInputGrad = head.propagate(cache, previous, headDelta);
//
//            gradInput = gradInput.add(headInputGrad);
//        }
//
//        return gradInput;
//    }

    public void compile(Random generator, WeightInitializer initializer) {
        for (AttentionHead head : heads) {
            head.compile(generator, initializer);
        }
    }

    public AttentionHead createAttentionHead() {
        return new AttentionHead(modelDimension, headDimension);
    }

    public Tensor attend(StatesCache cache, Tensor input, boolean training) {
        Tensor[] outputs = new Tensor[heads.size()];
        List<Thread> threads = new ArrayList<>();

        boolean shouldMultiThread = !training && heads.size() > 1;

        for (int i = 0; i < heads.size(); i++) {
            AttentionHead head = heads.get(i);
            int index = i;

            if (shouldMultiThread) {
                Thread thread = Thread.startVirtualThread(() -> outputs[index] = head.attend(cache, input));
                threads.add(thread);
            } else {
                outputs[i] = head.attend(cache, input);
            }
        }

        if (shouldMultiThread) {
            BrainUtils.waitAll(threads);
        }

        return TensorFactory.concat(List.of(outputs));
    }

    public int getTotalNeurons() {
        int total = 0;

        for (AttentionHead head : heads) {
            total += head.size();
        }

        return total;
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
