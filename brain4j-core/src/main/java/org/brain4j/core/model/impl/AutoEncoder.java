package org.brain4j.core.model.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.LayerNorm;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.core.training.evaluation.EvaluationResult;
import org.brain4j.math.Brain4JUtils;
import org.brain4j.math.Pair;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Autoencoder neural network model implementation. This class is an extension of the {@link Sequential} class.
 * <p>Due to the nature of auto encoders, accuracy and confusion matrix are not implemented, expect NaN and zeros.</p>
 */
public class AutoEncoder extends Sequential {

    private final int bottleNeckIndex;

    /**
     * Constructs an autoencoder.
     * @param bottleNeckIndex The index of the bottleneck layer, starting from 1.
     * @param layers The layers of the autoencoder, both encoder and decoder.
     */
    public AutoEncoder(int bottleNeckIndex, Layer... layers) {
        super(layers);
        this.bottleNeckIndex = bottleNeckIndex;
    }

    public EvaluationResult evaluate(ListDataSource dataSource) {
        int classes = dataSource.getSamples().getFirst().label().elements();

        // Binary classification
        if (classes == 1) {
            classes = 2;
        }

        List<Thread> threads = new ArrayList<>();
        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);

        dataSource.reset();

        while (dataSource.hasNext()) {
            Pair<Tensor, Tensor> partition = dataSource.nextBatch();
            threads.add(makeEvaluation(partition, null, totalLoss));
        }

        Brain4JUtils.waitAll(threads);

        return new EvaluationResult(totalLoss.get() / dataSource.size(), classes, new HashMap<>());
    }

    @Override
    public Thread makeEvaluation(Pair<Tensor, Tensor> batch, Map<Integer, Tensor> classifications, AtomicReference<Double> totalLoss) {
        return Thread.startVirtualThread(() -> {
            Tensor inputs = batch.first(); // [batch_size, input_size]
            Tensor expected = batch.second(); // [batch_size, output_size]

            Tensor prediction = predict(new StatesCache(), inputs, true); // [batch_size, output_size]
            int batchSize = inputs.shape()[0];

            for (int i = 0; i < batchSize; i++) {
                Range range = new Range(i, i + 1);

                Tensor output = prediction.slice(range).vector();
                Tensor target = expected.slice(range).vector();

                int predIndex = output.argmax();
                int targetIndex = target.argmax();

                if (output.elements() == 1) {
                    predIndex = output.get(0) > 0.5 ? 1 : 0;
                    targetIndex = (int) target.get(0);
                }

                double loss = lossFunction.calculate(target, output);
                totalLoss.updateAndGet(v -> v + loss);
            }
        });
    }

    @Override
    public Tensor predict(StatesCache cache, Tensor input, boolean training) {
        Layer workingLayer = layers.getFirst();

        if (input.dimension() == 1) {
            input = input.reshape(1, input.elements());
        }

        int elements = input.shape()[1];

        if (elements != workingLayer.getTotalNeurons()) {
            throw new IllegalArgumentException("Input dimensions do not match! (Input != Expected): %s != %s"
                    .formatted(elements, workingLayer.getTotalNeurons()));
        }

        Tensor denseResult = input.clone();

        cache.setInputTensor(workingLayer, denseResult);
        cache.setOutputTensor(workingLayer, denseResult);

        int size = training ? layers.size() : bottleNeckIndex;

        for (int l = 1; l < size; l++) {
            Layer layer = layers.get(l);

            cache.setInputTensor(layer, denseResult);

            denseResult = layer.forward(i, cache, denseResult, training);

            if (layer.canPropagate() && !(layer instanceof LayerNorm)) {
                workingLayer = layer;
            }
        }

        return denseResult;
    }

    public int getBottleNeckIndex() {
        return bottleNeckIndex;
    }
}
