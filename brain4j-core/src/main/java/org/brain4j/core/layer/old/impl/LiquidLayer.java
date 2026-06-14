package org.brain4j.core.layer.old.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.old.OldLayer;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.solver.NumericalSolver;
import org.brain4j.math.solver.impl.EulerSolver;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.commons.Range;
import org.brain4j.math.tensor.impl.GpuTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

/**
 * Liquid Time-Constant (LTC) recurrent layer.
 * <p>
 * This layer models continuous-time recurrent dynamics by evolving the hidden
 * state through a numerical ODE solver. Each hidden unit has a learnable
 * time constant {@code τ} constrained within a fixed range.
 * </p>
 *
 * <h2>Shape conventions:</h2>
 * <p>Input:</p>
 * <ul>
 *     <li>{@code input}: {@code [batch, timesteps, features]}</li>
 *     <li>{@code deltas}: {@code [batch, timesteps, features]}</li>
 * </ul>
 *
 * <p>Output:</p>
 * <ul>
 *     <li>{@code hidden}:
 *         <ul>
 *             <li>{@code [batch, timesteps, hidden_dim]} if {@code returnSequences = true}</li>
 *             <li>{@code [batch, hidden_dim]} otherwise</li>
 *         </ul>
 *     </li>
 *     <li>{@code deltas} is forwarded unchanged as second output</li>
 * </ul>
 *
 * <p>The hidden state dynamics follow the continuous-time equation:</p>
 * <blockquote><pre>
 * dh/dt = (-h + f(Wx + Uh + b)) / τ
 * </pre></blockquote>
 *
 * <p>where the integration over time is handled by a {@link NumericalSolver}
 * (e.g. Euler integration).</p>
 *
 * @implNote this layer expects exactly two input tensors: the signal and its time deltas
 * @author xEcho1337
 */
public class LiquidLayer extends OldLayer {
    
    private DenseLayer hiddenParams;
    private DenseLayer tauParams;
    private NumericalSolver solver;
    
    /* Hyperparameters */
    private int dimension;
    private double tauMin;
    private double tauMax;
    private boolean returnSequences;
    
    private LiquidLayer() {
    }
    
    /**
     * Creates a Liquid layer with default {@code τ} bounds and Euler integration.
     *
     * @param dimension       hidden state dimension
     * @param mSteps          number of solver steps per timestep
     * @param returnSequences whether to return the full hidden sequence
     */
    public LiquidLayer(int dimension, int mSteps, boolean returnSequences) {
        this(dimension, 0.5, 5.0, returnSequences, new EulerSolver(mSteps));
    }
    
    /**
     * Creates a Liquid layer with a custom numerical solver.
     *
     * @param dimension       hidden state dimension
     * @param returnSequences whether to return the full hidden sequence
     * @param solver          ODE solver used for state integration
     */
    public LiquidLayer(int dimension, boolean returnSequences, NumericalSolver solver) {
        this(dimension, 0.5, 5.0, returnSequences, solver);
    }
    
    /**
     * Creates a Liquid layer with explicit {@code τ} bounds.
     *
     * @param dimension       hidden state dimension
     * @param tauMin          minimum time constant
     * @param tauMax          maximum time constant
     * @param returnSequences whether to return the full hidden sequence
     * @param solver          ODE solver used for state integration
     */
    public LiquidLayer(int dimension, double tauMin, double tauMax,
                       boolean returnSequences, NumericalSolver solver) {
        this.solver = solver;
        this.dimension = dimension;
        this.tauMin = tauMin;
        this.tauMax = tauMax;
        this.returnSequences = returnSequences;
    }
    
    @Override
    public void connect() {
        this.weights = Tensors.zeros(previous.size(), dimension);
        this.bias = Tensors.zeros(dimension);
        this.hiddenParams = new DenseLayer(dimension);
        this.tauParams = new DenseLayer(dimension, Activations.SOFTPLUS);

        hiddenParams.connect();
        tauParams.connect();

    }
    
    @Override
    public void initWeights(RandomGenerator generator, int input, int output) {
        this.weights.map(x -> weightInit.generate(generator, input, output));
        this.hiddenParams.initWeights(generator, dimension, dimension);
        this.tauParams.initWeights(generator, input, dimension);
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        validateInputLength(inputs);
        
        Tensor input = inputs[0];
        Tensor deltas = inputs[1];
        
        if (input.rank() != 3) {
            throw Commons.illegalArgument(
                "Input must have shape [batch, timesteps, features]! Got: %s",
                Arrays.toString(input.shape()));
        }
        
        if (deltas.rank() != 3) {
            throw Commons.illegalArgument(
                "Deltas must have shape [batch, timesteps, features]! Got: %s",
                Arrays.toString(deltas.shape()));
        }
        
        int batch = input.shapeAt(0);
        int timesteps = input.shapeAt(1);
        
        Tensor hidden = Tensors.zeros(batch, dimension).withGrad();
        
        if (input instanceof GpuTensor gpu) {
            hidden = hidden.to(gpu.getDevice()).withGrad();
        }
        
        List<Tensor> hiddenStates = new ArrayList<>();
        
        Tensor projTau = tauParams.forward(cache, input)
            .map(v -> Commons.clamp(v, tauMin, tauMax));
        
        Tensor projInput = input.matmulGrad(weights).addGrad(bias);
        
        for (int t = 0; t < timesteps; t++) {
            Range[] ranges = { Range.all(), Range.point(t), Range.all() };
            
            Tensor deltaT = deltas.sliceGrad(ranges).squeezeGrad(1);
            Tensor tau_t = projTau.sliceGrad(ranges).squeezeGrad(1);
            Tensor projInput_t = projInput.sliceGrad(ranges).squeezeGrad(1);
            
            hidden = solver.update(
                deltaT, tau_t, projInput_t, hidden,
                x -> hiddenParams.forward(cache, x)
            );
            
            hiddenStates.add(hidden.reshapeGrad(batch, 1, dimension));
        }
        
        if (returnSequences) {
            hidden = Tensors.concatGrad(hiddenStates, 1);
        }
        
        return new Tensor[] { hidden, deltas };
    }
    
    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        super.backward(cache, updater, optimizer);
        hiddenParams.backward(cache, updater, optimizer);
        tauParams.backward(cache, updater, optimizer);
    }
    
    @Override
    public OldLayer freeze() {
        super.freeze();
        hiddenParams.freeze();
        tauParams.freeze();
        return this;
    }
    
    @Override
    public OldLayer unfreeze() {
        super.unfreeze();
        hiddenParams.unfreeze();
        tauParams.unfreeze();
        return this;
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dimension", dimension);
        object.addProperty("tau_min", tauMin);
        object.addProperty("tau_max", tauMax);
        object.addProperty("return_sequences", returnSequences);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.dimension = object.get("dimension").getAsInt();
        this.tauMin = object.get("tau_min").getAsFloat();
        this.tauMax = object.get("tau_max").getAsFloat();
        this.returnSequences = object.get("return_sequences").getAsBoolean();
    }
    
    @Override
    public void loadWeights(Map<String, Tensor> mappedWeights) {
        super.loadWeights(mappedWeights);
        tauParams.loadWeights(mappedWeights);
        hiddenParams.loadWeights(mappedWeights);
    }
    
    @Override
    public void toDevice(Device device) {
        super.toDevice(device);
        hiddenParams.toDevice(device);
        tauParams.toDevice(device);
    }
    
    @Override
    public void resetGrad() {
        super.resetGrad();
        hiddenParams.resetGrad();
        tauParams.resetGrad();
    }
    
    @Override
    public int size() {
        return dimension;
    }
    
    @Override
    public int getTotalWeights() {
        return weights.elements()
            + hiddenParams.getTotalWeights()
            + tauParams.getTotalWeights();
    }
    
    @Override
    public int getTotalBias() {
        return bias.elements()
            + hiddenParams.getTotalBias()
            + tauParams.getTotalBias();
    }
    
    @Override
    public Map<String, Tensor> weightsMap() {
        Map<String, Tensor> result = super.weightsMap();
        result.putAll(tauParams.weightsMap());
        result.putAll(hiddenParams.weightsMap());
        return result;
    }
    
    public DenseLayer getHiddenParams() {
        return hiddenParams;
    }
    
    public LiquidLayer setHiddenParams(DenseLayer hiddenParams) {
        this.hiddenParams = hiddenParams;
        return this;
    }
    
    public DenseLayer getTauParams() {
        return tauParams;
    }
    
    public LiquidLayer setTauParams(DenseLayer tauParams) {
        this.tauParams = tauParams;
        return this;
    }
    
    public NumericalSolver getSolver() {
        return solver;
    }
    
    public LiquidLayer setSolver(NumericalSolver solver) {
        this.solver = solver;
        return this;
    }
    
    public int getDimension() {
        return dimension;
    }
    
    public LiquidLayer setDimension(int dimension) {
        this.dimension = dimension;
        return this;
    }
    
    public double getTauMin() {
        return tauMin;
    }
    
    public LiquidLayer setTauMin(double tauMin) {
        this.tauMin = tauMin;
        return this;
    }
    
    public double getTauMax() {
        return tauMax;
    }
    
    public LiquidLayer setTauMax(double tauMax) {
        this.tauMax = tauMax;
        return this;
    }
    
    public boolean isReturnSequences() {
        return returnSequences;
    }
    
    public LiquidLayer setReturnSequences(boolean returnSequences) {
        this.returnSequences = returnSequences;
        return this;
    }
}
