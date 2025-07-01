package org.brain4j.core.importing;

import org.brain4j.core.importing.impl.OnnxLoader;
import org.brain4j.core.model.Model;

import java.nio.file.Files;
import java.nio.file.Paths;

public class ModelLoaders {
    
    public static Model fromOnnx(String path) throws Exception {
        byte[] data = Files.readAllBytes(Paths.get(path));
        
        OnnxLoader loader = new OnnxLoader();
        Model model = loader.deserialize(data);
        
        return model;
    }
}
