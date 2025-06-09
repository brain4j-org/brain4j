package org.brain4j.core.saving;

import com.github.luben.zstd.Zstd;
import org.brain4j.core.model.Model;

import java.io.File;

public interface ModelSerializer {

    default void serialize(String path, Model model) throws Exception {
        serialize(new File(path), model);
    }

    void serialize(File file, Model model) throws Exception;

    default Model deserialize(String path, Model model) throws Exception {
        return deserialize(new File(path), model);
    }

    Model deserialize(File file, Model model) throws Exception;

    static byte[] compress(byte[] data) {
        return Zstd.compress(data);
    }

    static byte[] decompress(byte[] data) {
        int size = (int) Zstd.getFrameContentSize(data);
        return Zstd.decompress(data, size);
    }
}
