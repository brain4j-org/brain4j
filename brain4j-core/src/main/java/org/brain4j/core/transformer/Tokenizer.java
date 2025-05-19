package org.brain4j.core.transformer;

import java.util.List;

public interface Tokenizer {
    List<String> tokenize(String input);
}
