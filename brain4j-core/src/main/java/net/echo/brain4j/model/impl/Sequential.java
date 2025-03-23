package net.echo.brain4j.model.impl;

import com.google.common.base.Preconditions;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.layer.impl.convolution.ConvLayer;
import net.echo.brain4j.layer.impl.convolution.FlattenLayer;
import net.echo.brain4j.layer.impl.convolution.InputLayer;
import net.echo.brain4j.layer.impl.convolution.PoolingLayer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.BackPropagation;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.math4j.BrainUtils;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Implementation of a sequential neural network model.
 * This model processes an input {@link Tensor} and produces an output {@link Tensor}.
 * It supports training using instances of {@link DataRow}.
 * </p>
 */
public class Sequential extends Model {

    public Sequential(Layer<?, ?>... layers) {
        super(layers);

        if (this.layers.isEmpty()) return;

        validateCNNIfPresent();
    }

    private void validateCNNIfPresent() {
        boolean isInput = layers.getFirst() instanceof InputLayer;
        boolean hasConv = layers.stream().anyMatch(Layer::isConvolutional);

        Preconditions.checkState(!(isInput && !hasConv), "Cannot use an input layer without convolutional layers!");
        Preconditions.checkState(!(!isInput && hasConv), "Cannot use a convolutional layer without an input layer!");
    }

    private Thread predictPartition(List<DataRow> partition, AtomicReference<Double> totalError) {
        return Thread.startVirtualThread(() -> {
            for (DataRow row : partition) {
                Tensor inputs = row.inputs();
                Tensor targets = row.outputs();

                Tensor outputs = predict(inputs);
                double loss = lossFunction.calculate(targets, outputs);

                totalError.updateAndGet(v -> v + loss);
            }
        });
    }

    private Thread makeEvaluation(List<DataRow> partition, Map<Integer, Tensor> classifications) {
        return Thread.startVirtualThread(() -> {
            for (DataRow row : partition) {
                Tensor prediction = predict(row.inputs());

                int predIndex = BrainUtils.indexOfMaxValue(prediction);
                int targetIndex = BrainUtils.indexOfMaxValue(row.outputs());

                if (row.outputs().dimension() == 1) {
                    predIndex = prediction.get(0) > 0.5 ? 1 : 0;
                    targetIndex = (int) row.outputs().get(0);
                }

                Tensor predictions = classifications.get(targetIndex);
                int pred = (int) predictions.get(predIndex);

                predictions.set(pred + 1, predIndex);
            }
        });
    }

    @Override
    public Sequential compile(LossFunctions function, Optimizer optimizer) {
        return compile(function.getFunction(), optimizer);
    }

    @Override
    public Sequential compile(LossFunction function, Optimizer optimizer) {
        return compile(WeightInit.UNIFORM_XAVIER.getFunction(), function, optimizer, new StochasticUpdater());
    }

    @Override
    public Sequential compile(WeightInitializer initializer, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        super.compile(initializer, lossFunction, optimizer, updater);

        connect(initializer, true);

        this.optimizer.postInitialize(this);
        this.updater.postInitialize(this);

        reloadWeights();
        return this;
    }

    @Override
    public Sequential compile(WeightInit initializer, LossFunctions lossFunction, Optimizer optimizer, Updater updater) {
        return compile(initializer.getFunction(), lossFunction.getFunction(), optimizer, updater);
    }

    @Override
    public EvaluationResult evaluate(DataSet<DataRow> dataSet) {
        int classes = dataSet.getData().getFirst().outputs().elements();

        // Binary classification
        if (classes == 1) {
            classes = 2;
        }

        Map<Integer, Tensor> classifications = new ConcurrentHashMap<>();

        for (int i = 0; i < classes; i++) {
            classifications.put(i, TensorFactory.create(classes));
        }

        List<Thread> threads = new ArrayList<>();

        if (!dataSet.isPartitioned()) {
            dataSet.partition(Math.min(Runtime.getRuntime().availableProcessors(), dataSet.getData().size()));
        }

        for (List<DataRow> partition : dataSet.getPartitions()) {
            threads.add(makeEvaluation(partition, classifications));
        }

        BrainUtils.waitAll(threads);
        return new EvaluationResult(classes, classifications);
    }

    @Override
    public double loss(DataSet<DataRow> dataSet) {
        reloadWeights();

        AtomicReference<Double> totalError = new AtomicReference<>(0.0);
        List<Thread> threads = new ArrayList<>();

        for (List<DataRow> partition : dataSet.getPartitions()) {
            threads.add(predictPartition(partition, totalError));
        }

        BrainUtils.waitAll(threads);
        return totalError.get() / dataSet.size();
    }

    @Override
    public void fit(DataSet<DataRow> dataSet) {
        propagation.iteration(dataSet);
    }

    @Override
    public Tensor predict(Tensor input) {
        return predict(new StatesCache(), input, false);
    }

    @Override
    public Tensor predict(StatesCache cache, Tensor input, boolean training) {
        Layer<?, ?> workingLayer = layers.getFirst();

        Preconditions.checkState(input.elements() == workingLayer.getTotalNeurons(), "Input dimension does not " +
                "match model input dimension! (Input != Expected " + input.elements() + " != " + workingLayer.getTotalNeurons() + ")");

        Kernel convolutionalResult = null;
        Tensor denseResult = input.clone();

        workingLayer.setInput(cache, denseResult);

        if (workingLayer instanceof InputLayer inputLayer) {
            convolutionalResult = inputLayer.getImage(cache);
        }

        for (int l = 1; l < layers.size(); l++) {
            Layer<?, ?> layer = layers.get(l);

            if (training && layer instanceof DropoutLayer) {
                layer.forward(cache, workingLayer, null);
                continue;
            }

            if (layer instanceof ConvLayer convLayer) {
                convolutionalResult = convLayer.forward(cache, workingLayer, convolutionalResult);
            }

            if (layer instanceof PoolingLayer poolingLayer) {
                convolutionalResult = poolingLayer.forward(cache, workingLayer, convolutionalResult);
            }

            if (layer instanceof FlattenLayer flattenLayer) {
                denseResult = flattenLayer.flatten(cache, workingLayer, convolutionalResult);
            }

            if (layer instanceof DenseLayer denseLayer) {
                denseResult = denseLayer.forward(cache, workingLayer, denseResult);
            }

            workingLayer = layer;
        }

        return denseResult;
    }

    @Override
    public void reloadWeights() {
        Layer<?, ?> lastLayer = layers.getFirst();

        for (int i = 1; i < layers.size(); i++) {
            Layer<?, ?> layer = layers.get(i);

            if (!(layer instanceof DenseLayer)) {
                lastLayer = null;
                continue;
            }

            if (lastLayer != null) {
                int input = lastLayer.getTotalNeurons();
                int output = layer.getTotalNeurons();

                Tensor weights = recalculateSynapseMatrix(lastLayer.getSynapses(), input, output);
                lastLayer.updateWeights(weights);
            }

            lastLayer = layer;
        }
    }

    /**
     * Recalculates the synapse matrix, used to cache the synapse weights for faster computation.
     *
     * @param synapses list of synapses to cache
     * @param inSize input size of the vector
     * @param outSize output size of the vector
     * @return the synapse matrix
     */
    public Tensor recalculateSynapseMatrix(List<Synapse> synapses, int inSize, int outSize) {
        Tensor weights = TensorFactory.matrix(outSize, inSize);

        for (int i = 0; i < outSize; i++) {
            for (int j = 0; j < inSize; j++) {
                Synapse synapse = synapses.get(j * outSize + i);
                weights.set(synapse.getWeight(), i, j);
            }
        }

        return weights;
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        stream.writeInt(layers.size());

        for (Layer<?, ?> layer : layers) {
            stream.writeUTF(layer.getClass().getName());
            layer.serialize(stream);
        }
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        int layersSize = stream.readInt();

        this.layers = new ArrayList<>();

        for (int i = 0; i < layersSize; i++) {
            String layerClassPath = stream.readUTF();
            Class<?> layerClass = Class.forName(layerClassPath);

            Constructor<?> constructor = layerClass.getDeclaredConstructor();
            constructor.setAccessible(true);

            Layer<?, ?> layer = (Layer<?, ?>) constructor.newInstance();
            layer.deserialize(stream);

            layers.add(layer);
        }
    }
}
