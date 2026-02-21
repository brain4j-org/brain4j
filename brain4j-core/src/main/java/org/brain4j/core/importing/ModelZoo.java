package org.brain4j.core.importing;

import org.brain4j.core.importing.format.impl.BrainAdapter;
import org.brain4j.core.importing.format.impl.OnnxAdapter;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.DAG;
import org.brain4j.core.model.impl.Sequential;

import java.io.File;

public class ModelZoo {
    
    public static final BrainAdapter BRAIN_FORMAT = new BrainAdapter();
    public static final OnnxAdapter ONNX_FORMAT = new OnnxAdapter();
    
    public static void saveModel(Sequential model, File file) {
        BRAIN_FORMAT.serialize(model, file);
    }
    
    public static void saveOnnx(DAG model, File file) {
        ONNX_FORMAT.serialize(model, file);
    }
    
    public static Sequential fromFile(String path) {
        return BRAIN_FORMAT.deserialize(new File(path));
    }
    
    public static DAG fromOnnx(String path) {
        return ONNX_FORMAT.deserialize(new File(path));
    }
}
