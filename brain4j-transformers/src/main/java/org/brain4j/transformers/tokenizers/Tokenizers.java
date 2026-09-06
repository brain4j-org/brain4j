package org.brain4j.transformers.tokenizers;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.brain4j.transformers.tokenizers.impl.BertPreTokenizer;
import org.brain4j.transformers.tokenizers.impl.BytePairTokenizer;
import org.brain4j.transformers.tokenizers.model.Tokenizer;

import java.io.*;

public class Tokenizers {
    
    public static final ObjectMapper MAPPER = new ObjectMapper();
    
    public static Tokenizer load(File file) throws IOException {
        if (!file.exists()) {
            throw new FileNotFoundException(file.getPath());
        }
        
        JsonNode root;
        
        try (Reader reader = new FileReader(file)) {
            root = MAPPER.readTree(reader);
        }
        
        if (root == null || !root.has("model")) {
            throw new IOException("Invalid tokenizer file: missing 'model' field");
        }
        
        JsonNode preTokenizer = root.get("pre_tokenizer");
        String tokenizerType = preTokenizer.get("type").asText();
        
        Tokenizer tokenizer = switch (tokenizerType) {
            case "ByteLevel" -> new BytePairTokenizer();
            case "BertPreTokenizer" -> new BertPreTokenizer();
            default -> throw new IOException("Unknown/unsupported pre-tokenizer type: " + tokenizerType);
        };
        
        tokenizer.load(file);
        return tokenizer;
    }
}
