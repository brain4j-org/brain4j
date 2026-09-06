package org.brain4j.transformers.tokenizers.model;

import org.brain4j.math.tensor.Tensor;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

// TODO: refactor tokenizers
public interface Tokenizer {
    
    List<String> splitTokens(String input);
    Tensor encode(List<String> tokens);
    String decode(int index);
    
    Map<String, Integer> getVocab();
    int vocabSize();
    int bosTokenId();
    int eosTokenId();
    void setBosTokenId(int id);
    void setEosTokenId(int id);
    
    void save(File file) throws IOException;
    void load(File file) throws IOException;
}
