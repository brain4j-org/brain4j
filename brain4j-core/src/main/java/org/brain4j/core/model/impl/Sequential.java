package org.brain4j.core.model.impl;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.Node;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.ModelBlock;
import org.brain4j.core.model.ModelLayout;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;

/**
 * Linear stack of layers evaluated in order.
 * Use it for feed-forward architectures built from {@link ModelLayout}.
 */
public record Sequential(Graph graph, ModelLayout layout, Device device, List<Layer> layers, int seed) implements Model, ModelBlock {

    public Sequential {
        if (layers.isEmpty()) {
            throw Commons.illegalArgument("Layer list is empty!");
        }
    }

    /**
     * Chains the layout layers and builds the backing graph.
     *
     * @param layout the compiled layout (not null)
     * @param device the target device ({@code null} for CPU)
     * @param seed the weight initialization seed
     */
    public Sequential(ModelLayout layout, Device device, int seed) {
        this(buildGraph(layout, seed), layout, device, layout.buildLayerList(), seed);
    }

    private static Graph buildGraph(ModelLayout layout, int seed) {
        Node lastNode = Node.input(layout.inputShape());

        for (Layer current : layout.buildLayerList()) {
            lastNode = current.apply(lastNode);
        }

        return Graph.of(seed, lastNode);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] predict(StatesCache cache, Tensor... inputs) {
        if (device != null && !cache.isTraining()) {
            device.createResources();
        }

        Tensor[] out = graph.predict(cache, inputs);

        if (device != null && !cache.isTraining()) {
            device.closeResources();
        }

        return out;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Sequential fork(Device device) {
        Graph newGraph = graph.fork(device);
        List<Layer> newLayers = new ArrayList<>(newGraph.layers());
        return new Sequential(newGraph, layout.copy(), device, newLayers, graph.seed());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Sequential to(Device device) {
        if (Objects.equals(this.device, device)) {
            return this;
        }

        for (Layer layer : layers) {
            layer.to(device);
        }

        return new Sequential(graph, layout, device, layers, seed);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Device device() {
        return device;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void summary() {
        Brain4J.fixConsole();

        StringBuilder stats = new StringBuilder();
        DecimalFormat format = new DecimalFormat("#,###");

        String pattern = "%-7s %-20s %-15s %-13s %-15s\n";
        String divider = Commons.getHeader(" Architecture ", Commons.HEADER_CHAR);

        stats.append(divider);
        stats.append(pattern.formatted("Index", "Layer Type", "Weights Shape", "Parameters", "Activation")).append("\n");
        stats.append(pattern.formatted("-", "Input", Arrays.toString(layout.inputShape().dims()), "0", "-")).append("\n");

        AtomicLong totalParams = new AtomicLong(0);
        AtomicLong trainableParams = new AtomicLong(0);

        append(pattern, stats, format, totalParams, trainableParams);

        byte floatSize = Float.BYTES;

        long totalParameters = totalParams.get();
        long trainableParameters = trainableParams.get();

        String sizeOfTotalParams = Commons.formatNumber(totalParams.get() * floatSize);
        String sizeOfTrainableParams = Commons.formatNumber(trainableParams.get() * floatSize);

        String formattedTotal = format.format(totalParameters);
        String formattedTrainable = format.format(trainableParameters);

        stats.append(Commons.getHeader(" Recap ", Commons.HEADER_CHAR));
        stats.append("Total parameters: %s (%s)\n".formatted(formattedTotal, sizeOfTotalParams));
        stats.append("Trainable parameters: %s (%s)\n".formatted(formattedTrainable, sizeOfTrainableParams));
        stats.append(Commons.getHeader("", Commons.HEADER_CHAR));

        Arrays.stream(stats.toString().split("\n")).forEach(System.out::println);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Sequential copy() {
        Graph newGraph = graph.copy();
        List<Layer> newLayers = new ArrayList<>(newGraph.layers());
        return new Sequential(newGraph, layout.copy(), device, newLayers, graph.seed());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void appendTo(List<Layer> layers) {
        layers.addAll(layers());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Layer> layers() {
        return Collections.unmodifiableList(layers);
    }

    /**
     * Returns the layout this model was compiled from.
     *
     * @return the model layout
     */
    public ModelLayout layout() {
        return layout;
    }

    /**
     * Returns the backing computation graph.
     *
     * @return the backing graph
     */
    public Graph graph() {
        return graph;
    }

    private void append(
        String pattern,
        StringBuilder builder,
        DecimalFormat format,
        AtomicLong totalParams,
        AtomicLong trainableParams
    ) {
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            String layerType = layer.getClass().getSimpleName();

            int total = layer.calculateTotalParameters();
            int trainable = layer.calculateTrainableParams();

            Tensor weightsTensor = layer.param("weights");

            String shape = weightsTensor == null ? "N/A" : Arrays.toString(weightsTensor.shape());
            String row = pattern.formatted(i, layerType, shape, format.format(total), layer.activation().name());

            builder.append(row);

            totalParams.addAndGet(total);
            trainableParams.addAndGet(trainable);
        }
    }
}
