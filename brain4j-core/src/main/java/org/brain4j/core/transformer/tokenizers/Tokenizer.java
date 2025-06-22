package org.brain4j.core.transformer.tokenizers;

import org.brain4j.math.tensor.Tensor;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public interface Tokenizer {
    
    default Tensor encode(String input) {
        return encodeTokens(splitTokens(input));
    }
    
    List<String> splitTokens(String input);

    Tensor encodeTokens(List<String> tokens);
    
    String decode(int index);

    Map<String, Integer> vocab();
    
    int vocabSize();

    void save(String path) throws IOException;

    void load(String path) throws IOException;
}
