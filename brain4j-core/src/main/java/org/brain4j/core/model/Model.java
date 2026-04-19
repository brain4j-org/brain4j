package org.brain4j.core.model;

import org.brain4j.core.importing.format.BinaryFormat;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.Copyable;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.commons.Range;
import org.brain4j.math.loss.LossFunction;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.loss.impl.BinaryCrossEntropy;
import org.brain4j.math.tensor.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Represents a generic neural network model.
 * <p>
 * A {@code Model} defines the forward computation logic, device placement
 * and structural introspection capabilities (layers and specifications),
 * without prescribing a specific training implementation.
 * </p>
 *
 * @author xEcho1337
 */
public interface Model<T> extends Copyable<Model<T>> {
    
    /**
     * Performs a full forward pass using a temporary {@link StatesCache}
     * and returns the first output tensor.
     * <p>
     * This is a convenience method for single-input, single-output models.
     * </p>
     *
     * @param inputs the input tensors
     * @return the first output tensor produced by the model
     */
    default Tensor predict(Tensor... inputs) {
        return predict(new StatesCache(), inputs)[0];
    }

    /**
     * Performs a full forward pass on the model using the provided cache.
     * <p>
     * The cache is used to store intermediate states required by certain
     * layers (e.g. for training or recurrent architectures).
     * </p>
     *
     * @param cache  the cache used during this forward pass
     * @param inputs one or more input tensors
     * @return an array containing all output tensors of the model
     */
    Tensor[] predict(StatesCache cache, Tensor... inputs);
    
    /**
     * Returns the device on which the model parameters are currently stored.
     * @return the device associated with this model
     */
    SiliconDevice device();

    /**
     * Returns the default format to use when saving this type of model.
     * @return the default save format for this model
     */
    BinaryFormat<T> saveFormat();
    
    /**
     * Evaluates the model on the given dataset.
     * <p>
     * This method runs inference over the entire dataset and computes
     * task-specific evaluation metrics (e.g. accuracy, loss).
     * </p>
     *
     * @param dataSource the dataset to evaluate the model on
     * @param lossFunction the loss function to use
     * @return an {@link EvaluationResult} containing evaluation metrics
     */
    default EvaluationResult evaluate(ListDataSource dataSource, LossFunction lossFunction) {
        int classes = Math.max(2, dataSource.getSamples().getFirst().getLabel(0).elements());
        Map<Integer, Tensor> classifications = new HashMap<>();

        if (!lossFunction.isRegression()) {
            for (int i = 0; i < classes; i++) {
                classifications.put(i, Tensors.zeros(classes));
            }
        }

        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);

        dataSource.reset();

        while (dataSource.hasNext()) {
            Batch batch = dataSource.nextBatch();
            makeEvaluation(batch, classifications, totalLoss, lossFunction);
        }

        return new EvaluationResult(totalLoss.get() / dataSource.getSize(), classes, classifications);
    }
    
    /**
     * Calculates the average loss on the given dataset.
     * <p>
     * This method works similarly to {@link Model#evaluate},
     * but it's much less RAM consuming.
     * </p>
     * @param dataSource the dataset to evaluate the model on
     * @param lossFunction the loss function to use
     * @return a value representing the average loss on the entire dataset
     */
    default double loss(ListDataSource dataSource, LossFunction lossFunction) {
        Map<Integer, Tensor> classifications = new HashMap<>();
        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);

        dataSource.reset();

        while (dataSource.hasNext()) {
            Batch batch = dataSource.nextBatch();
            makeEvaluation(batch, classifications, totalLoss, lossFunction);
        }

        return totalLoss.get() / dataSource.getSize();
    }
    
    /**
     * Prints a formatted summary of the model architecture to the console.
     * <p>
     * The summary typically includes:
     * <ul>
     *   <li>Layer types and order</li>
     *   <li>Input and output shapes</li>
     *   <li>Number of parameters per layer</li>
     *   <li>Total number of trainable parameters</li>
     * </ul>
     * </p>
     *
     * @throws IllegalStateException if the model has not been properly initialized
     */
    void summary();
    
    Model<T> fork(SiliconDevice device);
    
    /**
     * Returns an immutable view of the layers composing this object, in order.
     * @return an unmodifiable list of layers
     */
    List<Layer> getLayers();

    private void makeEvaluation(
        Batch batch,
        Map<Integer, Tensor> classifications,
        AtomicReference<Double> totalLoss,
        LossFunction lossFunction
    ) {
        Tensor[] inputs = batch.getFirst();
        Tensor[] labels = batch.getSecond();

        SiliconDevice device = device();

        if (device != null) device.createResources();

        StatesCache cache = new StatesCache(false);
        Tensor[] outputs = predict(cache, inputs);

        for (Tensor input : inputs) {
            int batchSize = input.shapeAt(0);

            for (int i = 0; i < outputs.length; i++) {
                Tensor output = outputs[i].to(null); // GPU -> CPU
                Tensor label = labels[i].to(null);   // GPU -> CPU

                for (int b = 0; b < batchSize; b++) {
                    Range range = Range.point(b);

                    Tensor sampleOutput = output.slice(range).flatten();
                    Tensor sampleLabel = label.slice(range).flatten();

                    int predIndex = sampleOutput.argmax();
                    int targetIndex = sampleLabel.argmax();

                    if (sampleOutput.elements() == 1 && lossFunction instanceof BinaryCrossEntropy) {
                        predIndex = sampleOutput.get(0) > 0.5 ? 1 : 0;
                        targetIndex = (int) sampleLabel.get(0);
                    }

                    double loss = lossFunction.calculate(sampleLabel, sampleOutput);
                    totalLoss.updateAndGet(v -> v + loss);

                    Tensor predictions = classifications.get(targetIndex);

                    if (predictions != null) {
                        int pred = (int) predictions.get(predIndex);
                        predictions.set(pred + 1, predIndex);
                    }
                }
            }
        }

        if (device != null) device.closeResources();
    }
}
