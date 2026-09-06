package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Liquid Time-Constant (LTC) recurrent layer.
 *
 * <p>Per timestep the hidden state evolves as:</p>
 * <blockquote><pre>
 *     h += (dt / tau) * (tanh(W_in x + b_in + W_rec h + b_rec) - h)
 *     tau = tauMin + softplus(W_tau x + b_tau)
 * </pre></blockquote>
 *
 * <p>Inputs: {@code x [batch, timesteps, features]}, optionally
 * {@code deltas [batch, timesteps]} with per-step time gaps (defaults to 1).
 * Output: {@code [batch, timesteps, hidden]} if {@code returnSequences},
 * otherwise {@code [batch, hidden]}.
 */
public class LiquidLayer extends Layer {

    public record Config(int hiddenDimension, int solverSteps, double tauMin, boolean returnSequences) {}

    protected final Config config;

    public LiquidLayer(int hiddenDimension) {
        this(hiddenDimension, 6, 0.5, true);
    }

    public LiquidLayer(int hiddenDimension, int solverSteps, boolean returnSequences) {
        this(hiddenDimension, solverSteps, 0.5, returnSequences);
    }

    public LiquidLayer(int hiddenDimension, int solverSteps, double tauMin, boolean returnSequences) {
        this(new Config(hiddenDimension, solverSteps, tauMin, returnSequences));
    }

    public LiquidLayer(Config config) {
        this.config = config;
    }

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

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        int features = inputShapes.getFirst().last();
        int hidden = config.hiddenDimension;

        generateWeights("weights_in", rng, features, hidden);
        generateWeights("weights_rec", rng, hidden, hidden);
        generateWeights("weights_tau", rng, features, hidden);
    }

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

        Tensor wIn = getParam("weights_in");
        Tensor bIn = getParam("bias_in");
        Tensor wRec = getParam("weights_rec");
        Tensor bRec = getParam("bias_rec");
        Tensor wTau = getParam("weights_tau");
        Tensor bTau = getParam("bias_tau");

        var tanh = Activations.TANH.function();
        var softplus = Activations.SOFTPLUS.function();

        Tensor h = Tensors.zeros(batch, hidden).withGrad();

        if (input instanceof GpuTensor gpu) {
            h = h.to(gpu.getDevice()).withGrad();
        }

        List<Tensor> states = new ArrayList<>();

        for (int t = 0; t < timesteps; t++) {
            Tensor xt = input.sliceGrad(Range.all(), Range.point(t), Range.all()).squeezeGrad(1);

            // tau in (tauMin, +inf), fully differentiable (no hard clamp)
            Tensor tauAct = xt.matmulGrad(wTau).addGrad(bTau).activateGrad(softplus);
            Tensor tau = tauAct.addGrad(fullLike(xt, batch, hidden, (float) config.tauMin));

            Tensor proj = xt.matmulGrad(wIn).addGrad(bIn);

            // Per-step time gaps, shared across substeps ([B, 1], no grad needed).
            Tensor dtCol = deltas == null ? null
                : deltas.slice(Range.all(), Range.point(t));

            for (int s = 0; s < config.solverSteps; s++) {
                h = eulerStep(h, proj, tau, dtCol, batch, hidden, wRec, bRec, tanh);
            }

            if (config.returnSequences) {
                states.add(h.reshapeGrad(batch, 1, hidden));
            }
        }

        if (config.returnSequences) {
            return new Tensor[] { Tensors.concatGrad(states, 1) };
        }

        return new Tensor[] { h };
    }

    private Tensor eulerStep(Tensor h, Tensor proj, Tensor tau, Tensor dtCol,
                             int batch, int hidden, Tensor wRec, Tensor bRec,
                             org.brain4j.math.activation.Activation tanh) {
        Tensor numer = fullLike(h, batch, hidden, (float) (1.0 / config.solverSteps));

        if (dtCol != null) {
            numer = numer.times(dtCol);
        }

        Tensor step = numer.withGrad().divGrad(tau);

        Tensor hProj = h.matmulGrad(wRec).addGrad(bRec);
        Tensor z = proj.addGrad(hProj).activateGrad(tanh);
        Tensor dh = z.subGrad(h);

        return h.addGrad(step.mulGrad(dh));
    }

    private static Tensor fullLike(Tensor ref, int batch, int hidden, float value) {
        Tensor full = Tensors.zeros(batch, hidden);

        if (ref instanceof GpuTensor gpu) {
            full = full.to(gpu.getDevice());
        }

        return full.plus(value);
    }

    @Override
    public Layer copy() {
        LiquidLayer copy = new LiquidLayer(config);
        copyParameters(copy);
        return copy;
    }

    public Config config() {
        return config;
    }
}
