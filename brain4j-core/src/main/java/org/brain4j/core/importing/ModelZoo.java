package org.brain4j.core.importing;

import org.brain4j.core.model.impl.GraphModel;
import org.brain4j.core.importing.format.impl.BrainAdapter;
import org.brain4j.core.importing.format.impl.OnnxAdapter;
import org.brain4j.core.model.Model;

import java.io.File;

public class ModelZoo {
    
    public static final BrainAdapter BRAIN_FORMAT = new BrainAdapter();
    public static final OnnxAdapter ONNX_FORMAT = new OnnxAdapter();
    
    public static void saveModel(Model model, File file) {
        BRAIN_FORMAT.serialize(model, file);
    }
    
    public static void saveOnnx(Model model, File file) {
        ONNX_FORMAT.serialize(model, file);
    }
    
    public static Model fromFile(String path) {
        return BRAIN_FORMAT.deserialize(new File(path));
    }
    
    public static GraphModel fromOnnx(String path) {
        return ONNX_FORMAT.deserialize(new File(path));
    }
}
