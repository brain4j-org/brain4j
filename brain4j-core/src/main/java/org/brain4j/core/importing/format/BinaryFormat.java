package org.brain4j.core.importing.format;

import java.io.File;

public interface BinaryFormat<T> {
    T deserialize(File file);
    void serialize(T input, File file);
}
