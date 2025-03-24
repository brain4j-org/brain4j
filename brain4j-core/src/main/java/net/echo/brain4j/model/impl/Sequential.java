package net.echo.brain4j.model.impl;

import com.google.common.base.Preconditions;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.Optimizer;
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
        boolean isInput = false; // layers.getFirst() instanceof InputLayer;
        boolean hasConv = layers.stream().anyMatch(Layer::isConvolutional);

        Preconditions.checkState(!(isInput && !hasConv), "Cannot use an input layer without convolutional layers!");
        Preconditions.checkState(!(!isInput && hasConv), "Cannot use a convolutional layer without an input layer!");
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

        connect(initializer);

        this.optimizer.postInitialize(this);
        this.updater.postInitialize(this);

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
        propagation.partitionIfRequired(dataSet);

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

        Tensor denseResult = input.clone();

        cache.setInputTensor(workingLayer, denseResult);
        cache.setOutputTensor(workingLayer, denseResult);

        for (int l = 1; l < layers.size(); l++) {
            Layer<?, ?> layer = layers.get(l);

            cache.setInputTensor(layer, denseResult);

            if (training && layer instanceof DropoutLayer dropout) {
                dropout.forward(cache, workingLayer, denseResult);
                continue;
            }

            if (layer instanceof DenseLayer dense) {
                denseResult = dense.forward(cache, workingLayer, denseResult);
            }

            if (layer instanceof LayerNorm norm) {
                denseResult = norm.forward(cache, workingLayer, denseResult);
                continue;
            }

            if (denseResult.checkNaN()) {
                System.out.println("NaN at " + layer.getId());
                System.out.println(denseResult);
            }
            workingLayer = layer;
        }

        return denseResult;
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
