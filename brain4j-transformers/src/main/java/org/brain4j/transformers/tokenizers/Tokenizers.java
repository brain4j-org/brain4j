package org.brain4j.transformers.tokenizers;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import org.brain4j.transformers.tokenizers.impl.BertPreTokenizer;
import org.brain4j.transformers.tokenizers.impl.BytePairTokenizer;
import org.brain4j.transformers.tokenizers.model.Tokenizer;

import java.io.*;

public class Tokenizers {
    
    public static final Gson GSON = new Gson();
    
    public static Tokenizer load(File file) throws IOException {
        if (!file.exists()) {
            throw new FileNotFoundException(file.getPath());
        }
        
        JsonObject root;
        
        try (Reader reader = new FileReader(file)) {
            root = GSON.fromJson(reader, JsonObject.class);
        }
        
        if (root == null || !root.has("model")) {
            throw new IOException("Invalid tokenizer file: missing 'model' field");
        }
        
        JsonObject preTokenizer = root.getAsJsonObject("pre_tokenizer");
        String tokenizerType = preTokenizer.get("type").getAsString();
        
        Tokenizer tokenizer = switch (tokenizerType) {
            case "ByteLevel" -> new BytePairTokenizer();
            case "BertPreTokenizer" -> new BertPreTokenizer();
            default -> throw new IOException("Unknown/unsupported pre-tokenizer type: " + tokenizerType);
        };
        
        tokenizer.load(file);
        return tokenizer;
    }
}
