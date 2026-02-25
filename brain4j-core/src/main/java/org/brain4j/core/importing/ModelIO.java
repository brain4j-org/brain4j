package org.brain4j.core.importing;

import org.brain4j.core.importing.format.BinaryAdapter;
import org.brain4j.core.model.Model;

import java.io.File;

public class ModelIO {

    public static <T extends Model> void save(T model, BinaryAdapter<T> adapter, File path) {
        adapter.serialize(model, path);
    }


    public static <T extends Model> T load(BinaryAdapter<T> adapter, File path) {
        return adapter.deserialize(path);
    }
}
