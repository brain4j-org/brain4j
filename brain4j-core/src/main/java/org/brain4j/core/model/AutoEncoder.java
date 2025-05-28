package org.brain4j.core.model;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.EvaluationResult;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.Commons;
import org.brain4j.math.Pair;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.brain4j.math.constants.Constants.*;
import static org.brain4j.math.constants.Constants.LIGHT_GREEN;
import static org.brain4j.math.constants.Constants.RESET;

/**
 * Autoencoder neural network model implementation. This class is an extension of the {@link Model} class.
 * <p>Due to the nature of auto encoders, accuracy and confusion matrix are not implemented, expect NaN and zeros.</p>
 */
public class AutoEncoder extends Model {

    private final int bottleNeckIndex;

    /**
     * Constructs an autoencoder.
     * @param bottleNeckIndex the index of the bottleneck layer, starting from 1
     * @param layers the layers of the autoencoder, both encoder and decoder
     */
    public AutoEncoder(int bottleNeckIndex, Layer... layers) {
        super(layers);
        this.bottleNeckIndex = bottleNeckIndex;
    }

    public EvaluationResult evaluate(ListDataSource dataSource) {
        int classes = dataSource.samples().getFirst().label().elements();

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

        Commons.waitAll(threads);

        return new EvaluationResult(totalLoss.get() / dataSource.size(), classes, new HashMap<>());
    }

    @Override
    public Thread makeEvaluation(Pair<Tensor, Tensor> batch, Map<Integer, Tensor> classifications, AtomicReference<Double> totalLoss) {
        return Thread.startVirtualThread(() -> {
            Tensor inputs = batch.first(); // [batch_size, input_size]
            Tensor expected = batch.second(); // [batch_size, output_size]

            Tensor prediction = predict(new StatesCache(this), inputs, true); // [batch_size, output_size]
            int batchSize = inputs.shape()[0];

            for (int i = 0; i < batchSize; i++) {
                Range range = new Range(i, i + 1);

                Tensor output = prediction.slice(range).vector();
                Tensor target = expected.slice(range).vector();

                double loss = lossFunction().calculate(target, output);
                totalLoss.updateAndGet(v -> v + loss);
            }
        });
    }

    @Override
    public Tensor predict(StatesCache cache, Tensor input, boolean training) {
        if (input == null || input.dimension() == 0) {
            throw new IllegalArgumentException("Input is either null or has dimension of 0!");
        }

        if (input.dimension() < 2) {
            // Shape: [batch_size, input_size]
            input = input.reshape(1, input.elements());
        }

        int[] shape = input.shape();

        Layer inputLayer = layers.getFirst();

        if (inputLayer.size() != shape[1]) {
            throw new IllegalArgumentException(
                "Input shape does not match. Expected: " + inputLayer.size() + ", Received: " + shape[1]
            );
        }

        Tensor pass = input;

        cache.setInput(0, input);
        cache.setOutput(0, pass);

        int size = training ? size() : bottleNeckIndex;

        for (int i = 0; i < size; i++) {
            Layer layer = layerAt(i);

            if (layer == null) {
                throw new IllegalStateException("Layer at index " + i + " is null!");
            }

            if (layer.skipPropagate()) continue;

            pass = layer.forward(new ForwardContext(cache, pass, i, training));
        }

        return pass;
    }

    @Override
    protected void printEvaluation(int step, int epoches, ListDataSource testSource) {
        EvaluationResult result = evaluate(testSource.clone());

        String lossMsg = "Loss: " + MAGENTA + "%.4f" + RESET;
        String accuracyMsg = "Accuracy: " + LIGHT_BLUE + "N/A" + RESET;
        String f1ScoreMsg = "F1-Score: " + LIGHT_GREEN + "N/A" + RESET;

        String message = "[%s/%s] " + lossMsg + " | " + accuracyMsg + " | " + f1ScoreMsg + "\n";
        System.out.printf(message, step, epoches, result.loss(), result.accuracy() * 100, result.f1Score() * 100);
    }

    public int bottleNeckIndex() {
        return bottleNeckIndex;
    }
}
