package org.brain4j.core.transformer;

import java.io.IOException;
import java.util.List;

public interface Tokenizer {
    List<String> tokenize(String input);

    void save(String path) throws IOException;

    void load(String path) throws IOException;
}
