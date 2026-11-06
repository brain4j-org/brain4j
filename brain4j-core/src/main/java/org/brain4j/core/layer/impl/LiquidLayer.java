package org.brain4j.core.layer.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * Integrates a liquid time-constant recurrent state over time steps.
 * Use it for irregularly sampled sequences where adaptive dynamics help.
 */
public class LiquidLayer extends Layer {

    /**
     * Holds hidden size, solver steps, minimum time constant and output mode.
     */
    public record Config(int hiddenDimension, int solverSteps, double tauMin, boolean returnSequences) {}

    protected final Config config;

    /**
     * Create a layer with default solver settings.
     */
    public LiquidLayer(int hiddenDimension) {
        this(hiddenDimension, 6, 0.5, true);
    }

    /**
     * Create a layer with the given solver steps and output mode.
     */
    public LiquidLayer(int hiddenDimension, int solverSteps, boolean returnSequences) {
        this(hiddenDimension, solverSteps, 0.5, returnSequences);
    }

    /**
     * Create a layer with the given solver steps, time floor and output mode.
     */
    public LiquidLayer(int hiddenDimension, int solverSteps, double tauMin, boolean returnSequences) {
        this(new Config(hiddenDimension, solverSteps, tauMin, returnSequences));
    }

    /**
     * Create a layer with the given config.
     */
    public LiquidLayer(Config config) {
        this.config = config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void build(List<Shape> inputShapes) {
        int features = inputShapes.getFirst().last();
        int hidden = config.hiddenDimension;

        registerParam("weights_in", Tensors.zeros(features, hidden));
        registerParam("bias_in", Tensors.zeros(hidden));
        registerParam("weights_rec", Tensors.zeros(hidden, hidden));
        registerParam("bias_rec", Tensors.zeros(hidden));
        registerParam("weights_tau", Tensors.zeros(features, hidden));
        registerParam("bias_tau", Tensors.zeros(hidden));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        int features = inputShapes.getFirst().last();
        int hidden = config.hiddenDimension;

        generateWeights("weights_in", rng, features, hidden);
        generateWeights("weights_rec", rng, hidden, hidden);
        generateWeights("weights_tau", rng, features, hidden);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        Shape first = inputShapes.getFirst();

        if (first.rank() != 2) {
            throw Commons.illegalArgument("Liquid expects rank-2 inputs [timesteps, features]! Got: %s",
                Arrays.toString(first.dims()));
        }

        int timesteps = first.dim(0);

        if (config.returnSequences) {
            return List.of(Shape.of(timesteps, config.hiddenDimension));
        }

        return List.of(Shape.of(config.hiddenDimension));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        if (inputs.length < 1 || inputs.length > 2) {
            throw Commons.illegalArgument("Liquid expects 1 input (signal) plus an optional deltas input! Got: %s",
                inputs.length);
        }

        Tensor input = inputs[0];

        if (input.rank() != 3) {
            throw Commons.illegalArgument("Signal must have shape [batch, timesteps, features]! Got: %s",
                Arrays.toString(input.shape()));
        }

        Tensor deltas = inputs.length > 1 ? inputs[1] : null;

        if (deltas != null && deltas.rank() != 2) {
            throw Commons.illegalArgument("Deltas must have shape [batch, timesteps]! Got: %s",
                Arrays.toString(deltas.shape()));
        }

        int batch = input.shapeAt(0);
        int timesteps = input.shapeAt(1);
        int hidden = config.hiddenDimension;

        Tensor wIn = param("weights_in");
        Tensor bIn = param("bias_in");
        Tensor wRec = param("weights_rec");
        Tensor bRec = param("bias_rec");
        Tensor wTau = param("weights_tau");
        Tensor bTau = param("bias_tau");

        var tanh = Activations.TANH.function();
        var softplus = Activations.SOFTPLUS.function();

        Tensor h = Tensors.zeros(batch, hidden).withGrad();

        if (input instanceof GpuTensor gpu) {
            h = h.to(gpu.device()).withGrad();
        }

        List<Tensor> states = new ArrayList<>();

        for (int t = 0; t < timesteps; t++) {
            Tensor xt = input.slice(Range.all(), Range.point(t), Range.all()).squeeze(1);

            Tensor tauAct = xt.matmul(wTau).add(bTau).activate(softplus);
            Tensor tau = tauAct.add(fullLike(xt, batch, hidden, (float) config.tauMin));

            Tensor proj = xt.matmul(wIn).add(bIn);

            Tensor dtCol = deltas == null ? null
                : deltas.slice(Range.all(), Range.point(t));

            for (int s = 0; s < config.solverSteps; s++) {
                h = eulerStep(h, proj, tau, dtCol, batch, hidden, wRec, bRec, tanh);
            }

            if (config.returnSequences) {
                states.add(h.reshape(batch, 1, hidden));
            }
        }

        if (config.returnSequences) {
            return new Tensor[] { Tensors.concat(states, 1) };
        }

        return new Tensor[] { h };
    }

    private Tensor eulerStep(Tensor h, Tensor proj, Tensor tau, Tensor dtCol,
                             int batch, int hidden, Tensor wRec, Tensor bRec,
                             Activation tanh) {
        Tensor numer = fullLike(h, batch, hidden, (float) (1.0 / config.solverSteps));

        if (dtCol != null) {
            numer = numer.times(dtCol);
        }

        Tensor step = numer.withGrad().div(tau);

        Tensor hProj = h.matmul(wRec).add(bRec);
        Tensor z = proj.add(hProj).activate(tanh);
        Tensor dh = z.sub(h);

        return h.add(step.mul(dh));
    }

    private static Tensor fullLike(Tensor ref, int batch, int hidden, float value) {
        Tensor full = Tensors.zeros(batch, hidden);

        if (ref instanceof GpuTensor gpu) {
            full = full.to(gpu.device());
        }

        return full.plus(value);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        LiquidLayer copy = new LiquidLayer(config);
        copyParameters(copy);

        return copy;
    }

    /**
     * Return the layer config.
     *
     * @return the config
     */
    public Config config() {
        return config;
    }
}
