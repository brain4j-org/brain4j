package org.brain4j.core.importing.io;

import org.brain4j.core.importing.format.BinaryFormat;
import org.brain4j.core.model.Model;

import java.io.File;

public class ModelIO {

    public static <T extends Model> void save(T model, BinaryFormat<T> adapter, File path) {
        adapter.serialize(model, path);
    }
    
    public static <T extends Model> T load(BinaryFormat<T> adapter, File path) {
        return adapter.deserialize(path);
    }
}
