package org.brain4j.core.model.impl;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.Pair;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.brain4j.math.constants.Constants.*;

/**
 * Autoencoder neural network model implementation. This class is an extension of the {@link Sequential} class.
 * <p>Due to the nature of auto encoders, accuracy and confusion matrix are not implemented, expect NaN and zeros.</p>
 */
public class AutoEncoder extends Sequential {

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

        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);

        Pair<Tensor[], Tensor> data = dataSource.allData();
        makeEvaluation(data, null, totalLoss);

        return new EvaluationResult(totalLoss.get() / dataSource.size(), classes, new HashMap<>());
    }

    @Override
    public void makeEvaluation(Pair<Tensor[], Tensor> batch, Map<Integer, Tensor> classifications, AtomicReference<Double> totalLoss) {
        Tensor[] inputs = batch.first(); // [batch_size, input_size]
        Tensor expected = batch.second(); // [batch_size, output_size]

        Tensor prediction = predict(new StatesCache(deviceType), true, inputs).cpu(); // [batch_size, output_size]

        for (Tensor input : inputs) {
            int batchSize = input.shape()[0];

            for (int i = 0; i < batchSize; i++) {
                Range range = new Range(i, i + 1);

                Tensor output = prediction.slice(range).vector();
                Tensor target = expected.slice(range).vector();

                double loss = lossFunction().calculate(target, output);
                totalLoss.updateAndGet(v -> v + loss);
            }
        }
    }

    @Override
    public Tensor predict(StatesCache cache, boolean training, Tensor... inputs) {
        Tensor input = validateInputs(inputs);
        Tensor result = input.to(deviceType).withGrad();

        int size = training ? flattened.size() : bottleNeckIndex;

        for (int i = 0; i < size; i++) {
            Layer layer = flattenedAt(i);

            if (layer == null) {
                throw new IllegalStateException("Layer at index " + i + " is null!");
            }

            result = layer.forward(new ForwardContext(cache, result, i, training));
        }

        return result;
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
