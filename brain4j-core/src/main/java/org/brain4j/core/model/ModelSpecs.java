package org.brain4j.core.model;

import org.brain4j.core.layer.Layer0;
import org.brain4j.core.model.impl.Sequential;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ModelSpecs implements ModelBlock, Cloneable {
    
    private List<ModelBlock> components = new ArrayList<>();
    private boolean frozen = false;
    
    public static ModelSpecs of(ModelBlock... components) {
        if (components == null) {
            throw new IllegalArgumentException("Component list cannot be null!");
        }
        
        ModelSpecs specs = new ModelSpecs();
        specs.components.addAll(List.of(components));
        
        return specs;
    }
    
    @Override
    public void appendTo(List<Layer0> layers) {
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
    
    public Model compile() {
        return compile(System.currentTimeMillis());
    }
    
    public Model compile(long seed) {
        this.frozen = true;
        return new Sequential(this, null, seed);
    }
    
    public List<ModelBlock> getComponents() {
        if (frozen) {
            return Collections.unmodifiableList(components);
        }
        
        return components;
    }
    
    public List<Layer0> buildLayerList() {
        List<Layer0> flat = new ArrayList<>();
        appendTo(flat);
        return flat;
    }
    
    @Override
    public ModelSpecs clone() {
        try {
            ModelSpecs clone = (ModelSpecs) super.clone();
            
            clone.frozen = false;
            clone.components = new ArrayList<>();
            clone.components.addAll(components);
            
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();
        }
    }
}
