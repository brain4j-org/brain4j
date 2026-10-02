package org.brain4j.core.model;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.math.Copyable;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Shape;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ModelSpecs implements ModelBlock, Copyable<ModelSpecs> {
    
    private final Shape inputShape;
    private final List<ModelBlock> components = new ArrayList<>();
    private boolean frozen = false;

    private ModelSpecs(Shape inputShape) {
        if (inputShape == null) {
            throw Commons.illegalArgument("Input shape must not be null!");
        }

        this.inputShape = inputShape;
    }

    public static ModelSpecs of(Shape inputShape, List<ModelBlock> components) {
        if (components == null) {
            throw Commons.illegalArgument("Component list must not be null!");
        }

        ModelSpecs specs = new ModelSpecs(inputShape);
        specs.components.addAll(components);

        return specs;
    }

    public static ModelSpecs of(Shape inputShape, ModelBlock... components) {
        if (components == null) {
            throw Commons.illegalArgument("Component list must not be null!");
        }

        return of(inputShape, List.of(components));
    }

    public Shape inputShape() {
        return inputShape;
    }

    @Override
    public void appendTo(List<Layer> layers) {
        for (ModelBlock component : components) {
            component.appendTo(layers);
        }
    }
    
    public ModelSpecs add(ModelBlock component) {
        if (frozen) {
            throw new IllegalArgumentException("ModelSpecs has been compiled and cannot be modified! Consider checking out clone().");
        }

        if (component == null) {
            throw Commons.illegalArgument("Component must not be null!");
        }

        components.add(component);
        return this;
    }
    
    public Sequential compile() {
        return compile((int) (System.currentTimeMillis() % 1_000_000_000));
    }
    
    public Sequential compile(int seed) {
        this.frozen = true;
        return new Sequential(this, null, seed);
    }
    
    public List<ModelBlock> getComponents() {
        if (frozen) {
            return Collections.unmodifiableList(components);
        }
        
        return components;
    }
    
    public List<Layer> buildLayerList() {
        List<Layer> flat = new ArrayList<>();
        appendTo(flat);
        return flat;
    }
    
    @Override
    public ModelSpecs copy() {
        ModelSpecs copy = new ModelSpecs(inputShape.copy());
        copy.components.addAll(components);
        return copy;
    }
}
