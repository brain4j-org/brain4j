package org.brain4j.core.model;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.loss.LossFunction;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

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
public interface Model extends ModelBlock {
    
    /**
     * Performs a full forward pass using a temporary {@link StatesCache}
     * and returns the first output tensor.
     * <p>
     * This is a convenience method for single-input, single-output models.
     * </p>
     *
     * @param input the input tensor
     * @return the first output tensor produced by the model
     */
    default Tensor predict(Tensor input) {
        return predict(new StatesCache(false), input)[0];
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
    EvaluationResult evaluate(ListDataSource dataSource, LossFunction lossFunction);
    
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
    double loss(ListDataSource dataSource, LossFunction lossFunction);
    
    /**
     * Copies all model parameters to the specified device.
     *
     * @param device the target device
     * @return a copy of this model instance
     */
    Model fork(SiliconDevice device);
    
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
    
    /**
     * Returns the specifications used to construct this model.
     * <p>
     * {@link ModelSpecs} describes the logical structure of the model
     * independently of its runtime state.
     * </p>
     *
     * @return the model specifications
     */
    ModelSpecs getSpecs();

    /**
     * Returns the device on which the model parameters are currently stored.
     * @return the device associated with this model
     */
    SiliconDevice getDevice();

    /**
     * Returns an immutable view of the layers composing this model, in execution order.
     * @return an unmodifiable list of layers
     */
    List<Layer> getLayers();

    @Override
    default void appendTo(List<Layer> layers) {
        layers.addAll(getLayers());
    }
}
