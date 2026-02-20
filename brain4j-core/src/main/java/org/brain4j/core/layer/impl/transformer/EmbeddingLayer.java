package org.brain4j.core.layer.impl.transformer;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer0;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

/**
 * Embedding layer implementation for transformer architectures.
 * <p>
 * This layer maps integer token indices to dense embedding vectors.
 * It expects an input tensor of shape <code>[batch, seq_len]</code> where
 * each element is a token ID located in the vocabulary.
 * </p>
 * <p>
 * The output is a tensor of shape <code>[batch, seq_len, embedding_dim]</code>,
 * where each token index is replaced by its corresponding embedding vector.
 * </p>
 * @author xEcho1337
 */
public class EmbeddingLayer extends Layer0 {

    private int vocabSize;
    private int embeddingDim;
    
    public EmbeddingLayer() {
    }
    
    /**
     * Constructs a new instance of an embedding layer.
     * @param vocabSize the vocabulary size
     * @param embeddingDim the embedding dimension
     */
    public EmbeddingLayer(int vocabSize, int embeddingDim) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.weightInit = new UniformXavierInit();
    }
    
    @Override
    public void connect() {
        this.weights = Tensors.zeros(vocabSize, embeddingDim).withGrad();
    }
    
    @Override
    public void initWeights(RandomGenerator generator, int input, int output) {
        this.weights.map(x -> weightInit.generate(generator, input, output));
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        validateInputLength(inputs);

        Tensor input = inputs[0];
        validateInputTensor(input, "Input must have shape [batch, tokens]! Got: %s", Arrays.toString(input.shape()));

        int[] shape = input.shape();

        if (shape.length != 2) {
            throw Commons.illegalState("Input must have shape [batch, seq_length]! Got: %s", Arrays.toString(shape));
        }

        int batchSize = shape[0];
        int seqLength = shape[1];
        
        Tensor output = Tensors.zeros(batchSize, seqLength, embeddingDim);
        
        if (input.usesGrad()) output = output.withGrad();

        float[] outData = output.data();
        float[] weightData = weights.data();
        float[] inputData = input.data();
        
        IntStream.range(0, batchSize).parallel().forEach(b -> {
            for (int s = 0; s < seqLength; s++) {
                int index = input.linearIndex(b, s);
                int tokenId = (int) inputData[index];
                int outOffset = (b * seqLength + s) * embeddingDim;
                int weightOffset = tokenId * embeddingDim;

                System.arraycopy(weightData, weightOffset, outData, outOffset, embeddingDim);
            }
        });

        if (input instanceof GpuTensor gpuInput) {
            output = output.to(gpuInput.device());
        }

        cache.setStates(this, "input", inputs);
        cache.setStates(this, "output", output);

        // [batch, seq_len, embedding_dim]
        return new Tensor[] { output };
    }
    
    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        if (!weights.usesGrad()) return;
        
        Tensor input = cache.getStates(this, "input")[0];
        Tensor output = cache.getStates(this, "output")[0];
        Tensor gradOutput = output.grad();
        
        int[] shape = output.shape();
        
        int batchSize = shape[0];
        int seqLength = shape[1];
        
        Tensor weightsGrad = weights.grad();
        
        if (weightsGrad == null) {
            weightsGrad = Tensors.zeros(weights.shape());
        }
        
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLength; s++) {
                int tokenId = (int) input.get(b, s);
                
                for (int d = 0; d < embeddingDim; d++) {
                    float gradient = gradOutput.get(b, s, d);
                    weightsGrad.set(gradient, tokenId, d);
                }
            }
        }
        
        Tensor optimized = optimizer.step(weights, weightsGrad);
        
        clipper.clip(optimized);
        updater.change(weights, optimized);
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("vocab_size", vocabSize);
        object.addProperty("embedding_dim", embeddingDim);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.vocabSize =  object.get("vocab_size").getAsInt();
        this.embeddingDim = object.get("embedding_dim").getAsInt();
    }
    
    @Override
    public boolean validInput(Tensor input) {
        return input.rank() == 2;
    }
    
    @Override
    public int size() {
        return embeddingDim;
    }
}
