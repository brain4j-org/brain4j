package org.brain4j.core.adapters;

import com.github.luben.zstd.Zstd;
import org.brain4j.core.model.Model;

import java.io.File;

public interface ModelAdapter {

    void serialize(String path, Model model) throws Exception;

    void serialize(File file, Model model) throws Exception;

    Model deserialize(String path, Model model) throws Exception;

    Model deserialize(File file, Model model) throws Exception;

    static byte[] compress(byte[] data) {
        return Zstd.compress(data);
    }

    static byte[] decompress(byte[] data) {
        int size = (int) Zstd.getFrameContentSize(data);
        return Zstd.decompress(data, size);
    }
}
