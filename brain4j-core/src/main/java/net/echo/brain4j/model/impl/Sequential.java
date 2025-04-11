package net.echo.brain4j.model.impl;

import com.google.common.base.Preconditions;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.layer.impl.conv.InputLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.math4j.BrainUtils;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Implementation of a sequential neural network model.
 * This model processes an input {@link Tensor} and produces an output {@link Tensor}.
 * It supports training using instances of {@link DataRow}.
 * </p>
 */
public class Sequential extends Model {

    public Sequential(Layer... layers) {
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

    @Override
    public Thread makeEvaluation(List<DataRow> partition, Map<Integer, Tensor> classifications, AtomicReference<Double> totalLoss) {
        return Thread.startVirtualThread(() -> {
            for (DataRow row : partition) {
                Tensor prediction = predict(row.inputs());
                Tensor expected = row.outputs();

                int predIndex = BrainUtils.argmax(prediction);
                int targetIndex = BrainUtils.argmax(expected);

                if (row.outputs().elements() == 1) {
                    predIndex = prediction.get(0) > 0.5 ? 1 : 0;
                    targetIndex = (int) row.outputs().get(0);
                }

                double loss = lossFunction.calculate(expected, prediction);
                totalLoss.updateAndGet(v -> v + loss);

                Tensor predictions = classifications.get(targetIndex);
                int pred = (int) predictions.get(predIndex);

                predictions.set(pred + 1, predIndex);
            }
        });
    }

    @Override
    public Sequential compile(Loss function, Optimizer optimizer) {
        return compile(function.getFunction(), optimizer);
    }

    @Override
    public Sequential compile(LossFunction function, Optimizer optimizer) {
        return compile(WeightInit.UNIFORM_XAVIER.getFunction(), function, optimizer, new StochasticUpdater());
    }

    @Override
    public Sequential compile(WeightInitializer initializer, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        super.compile(initializer, lossFunction, optimizer, updater);

        connect(initializer);

        this.optimizer.postInitialize(this);
        this.updater.postInitialize();

        return this;
    }

    @Override
    public Sequential compile(WeightInit initializer, Loss lossFunction, Optimizer optimizer, Updater updater) {
        return compile(initializer.getFunction(), lossFunction.getFunction(), optimizer, updater);
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
        Layer workingLayer = layers.getFirst();

        if (input.elements() != workingLayer.getTotalNeurons()) {
            throw new IllegalArgumentException("Input dimensions do not match! (Input != Expected): %s != %s"
                    .formatted(input.elements(), workingLayer.getTotalNeurons()));
        }

        Tensor denseResult = input.clone();

        cache.setInputTensor(workingLayer, denseResult);
        cache.setOutputTensor(workingLayer, denseResult);

        for (int l = 1; l < layers.size(); l++) {
            Layer layer = layers.get(l);

            cache.setInputTensor(layer, denseResult);

            Layer nextLayer = l < layers.size() - 1 ? layers.get(l + 1) : null;
            denseResult = layer.forward(cache, workingLayer, nextLayer, denseResult, training);

            if (layer.canPropagate()) {
                workingLayer = layer;
            }
        }

        return denseResult;
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        stream.writeInt(layers.size());

        for (Layer layer : layers) {
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

            Layer layer = (Layer) constructor.newInstance();
            layer.deserialize(stream);

            layers.add(layer);
        }
    }
}
