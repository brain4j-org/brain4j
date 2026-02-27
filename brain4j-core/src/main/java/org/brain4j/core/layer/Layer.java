package org.brain4j.core.layer;

import org.brain4j.core.model.ModelBlock;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Copyable;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.clipper.impl.HardClipper;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.WeightInit;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

public abstract class Layer implements Copyable<Layer>, ModelBlock {
    
    protected Map<String, Tensor> parameters;
    protected GradientClipper clipper;
    protected Activation activation;
    protected WeightInit weightInit;
    protected boolean frozen;
    
    public Layer() {
        this(new Linear());
    }
    
    public Layer(Activation activation) {
        this.parameters = new HashMap<>();
        this.clipper = new HardClipper(5);
        this.activation = activation;
        this.weightInit = activation.defaultWeightInit();
    }
    
    @Override
    public void appendTo(List<Layer> layers) {
        layers.add(this);
    }
    
    public abstract void build(List<Shape> inputShapes);
    
    public abstract void initWeights(List<Shape> inputShapes, RandomGenerator rng);
    
    public abstract List<Shape> inferOutputShapes(List<Shape> inputShapes);
    
    public abstract Tensor[] forward(StatesCache cache, Tensor... inputs);
    
    protected Tensor[] tensors(Tensor... values) {
        return values;
    }
    
    public void initAutoGrad() {
        for (Tensor param : parameters.values()) {
            param.withGrad();
        }
    }

    public void generateWeights(String id, RandomGenerator rng, int input, int output) {
        Tensor param = parameters.get(id);
        
        if (param == null) {
            throw Commons.illegalArgument("No parameter with id '%s' was found!", id);
        }
        
        param.map(x -> weightInit.generate(rng, input, output));
    }
    
    public void copyParameters(Layer other) {
        Map<String, Tensor> newParameters = new HashMap<>();
        parameters.forEach((k, v) -> newParameters.put(k, v.copy()));
        
        other.parameters.clear();
        other.parameters.putAll(newParameters);

    }
    
    public void to(SiliconDevice device) {
        Map<String, Tensor> newParameters = new HashMap<>();
        parameters.forEach((k, v) -> newParameters.put(k, v.to(device)));
        
        parameters.clear();
        parameters.putAll(newParameters);
    }
    
    public Node apply(Node... inputs) {
        return new Node(this, List.of(inputs));
    }
    
    public Tensor getParam(String name) {
        return parameters.get(name);
    }
    
    public void resetGrad() {
        for (Tensor parameter : parameters.values()) {
            parameter.zeroGrad();
        }
    }
    
    public Layer freeze() {
        frozen = true;
        parameters.replaceAll((k, v) -> v.noGrad());
    
        return this;
    }
    
    public Layer unfreeze() {
        frozen = false;
        parameters.replaceAll((k, v) -> v.withGrad());

        return this;
    }
    
    public int calculateTrainableParams() {
        return parameters.values()
            .stream()
            .filter(Tensor::usesGrad)
            .mapToInt(Tensor::elements)
            .sum();
    }
    
    public int calculateTotalParameters() {
        return parameters.values()
            .stream()
            .mapToInt(Tensor::elements)
            .sum();
    }
    
    public Map<String, Tensor> parameters() {
        return parameters;
    }
    
    public GradientClipper clipper() {
        return clipper;
    }

    public Layer clipper(GradientClipper clipper) {
        this.clipper = clipper;
        return this;
    }

    public Activation activation() {
        return activation;
    }

    public Layer activation(Activation activation) {
        this.activation = activation;
        return this;
    }
    
    public WeightInit weightInit() {
        return weightInit;
    }

    public Layer weightInit(WeightInit weightInit) {
        this.weightInit = weightInit;
        return this;
    }
    
    public boolean frozen() {
        return frozen;
    }
    
    public void backward(Updater updater, Optimizer optimizer) {
        for (Tensor parameter : parameters.values()) {
            Tensor grad = parameter.grad();
            
            if (grad == null) continue;
            
            Tensor optimized = grad;
            
            if (optimized.rank() > parameter.rank()) {
                while (optimized.rank() > parameter.rank()) {
                    optimized = optimized.sum(0, false);
                }
            } else {
                optimized = optimizer.step(parameter, grad);
            }
            
            clipper.clip(optimized);
            updater.change(parameter, optimized);
        }
    }
}