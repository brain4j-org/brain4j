package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.Layer;
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

public class EmbeddingLayer extends Layer {

    public record Config(int vocabSize, int embeddingDim) {}

    protected Config config;

    private Tensor lastInput;
    private Tensor lastOutput;

    public EmbeddingLayer(int vocabSize, int embeddingDim) {
        this(new Config(vocabSize, embeddingDim));
    }

    public EmbeddingLayer(Config config) {
        this.config = config;
        this.weightInit = new UniformXavierInit();
    }

    @Override
    public void build(List<Shape> inputShapes) {
        registerParam("weights", Tensors.zeros(config.vocabSize, config.embeddingDim));
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        generateWeights("weights", rng, config.vocabSize, config.embeddingDim);
    }

    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires 1 input but %s were given!", inputShapes.size());
        }

        Shape inputShape = inputShapes.getFirst();

        if (inputShape.rank() != 1) {
            throw Commons.illegalArgument("Embedding requires tensors with rank 1 but %s were given!", inputShape.rank());
        }

        return List.of(Shape.of(inputShape.dim(0), config.embeddingDim));
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        int[] shape = input.shape();

        if (shape.length != 2) {
            throw Commons.illegalState("Input must have shape [batch, seq_length]! Got: %s",
                java.util.Arrays.toString(shape));
        }

        int batchSize = shape[0];
        int seqLength = shape[1];

        Tensor output = Tensors.zeros(batchSize, seqLength, config.embeddingDim);

        if (input.usesGrad()) output = output.withGrad();

        float[] outData = output.data();
        float[] weightData = getParam("weights").data();
        float[] inputData = input.data();

        IntStream.range(0, batchSize).parallel().forEach(b -> {
            for (int s = 0; s < seqLength; s++) {
                int index = input.linearIndex(b, s);
                int tokenId = (int) inputData[index];
                int outOffset = (b * seqLength + s) * config.embeddingDim;
                int weightOffset = tokenId * config.embeddingDim;

                System.arraycopy(weightData, weightOffset, outData, outOffset, config.embeddingDim);
            }
        });

        if (input instanceof GpuTensor gpuInput) {
            output = output.to(gpuInput.getDevice());
        }

        if (cache.isTraining()) {
            lastInput = input;
            lastOutput = output;
        }

        return new Tensor[] { output };
    }

    @Override
    public void backward(Updater updater, Optimizer optimizer) {
        Tensor weights = getParam("weights");

        if (!weights.usesGrad()) return;
        if (lastInput == null || lastOutput == null) return;

        Tensor gradOutput = lastOutput.grad();

        if (gradOutput == null) return;

        int[] shape = lastOutput.shape();
        int batchSize = shape[0];
        int seqLength = shape[1];

        Tensor weightsGrad = weights.grad();

        if (weightsGrad == null) {
            weightsGrad = Tensors.zeros(weights.shape());
        }

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLength; s++) {
                int tokenId = (int) lastInput.get(b, s);

                for (int d = 0; d < config.embeddingDim; d++) {
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
    public Layer copy() {
        EmbeddingLayer copy = new EmbeddingLayer(config);
        copyParameters(copy);
        return copy;
    }

    public Config config() {
        return config;
    }
}
