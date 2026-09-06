package org.brain4j.core.model;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.math.Copyable;
import org.brain4j.math.commons.Commons;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ModelSpecs implements ModelBlock, Copyable<ModelSpecs> {
    
    private final List<ModelBlock> components = new ArrayList<>();
    private boolean frozen = false;
    
    public static ModelSpecs of(List<ModelBlock> components) {
        if (components == null) {
            throw Commons.illegalArgument("Component list must not be null!");
        }
        
        ModelSpecs specs = new ModelSpecs();
        specs.components.addAll(components);
        
        return specs;
    }
    
    public static ModelSpecs of(ModelBlock... components) {
        return of(List.of(components));
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
        return ModelSpecs.of(components);
    }
}
