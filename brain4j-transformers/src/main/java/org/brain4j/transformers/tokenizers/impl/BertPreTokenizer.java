package org.brain4j.transformers.tokenizers.impl;

import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;
import org.brain4j.transformers.tokenizers.model.AddedToken;
import org.brain4j.transformers.tokenizers.model.Normalizer;

import java.io.*;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

public class BertPreTokenizer extends BytePairTokenizer {
    
    public BertPreTokenizer() {
        super("##");
    }
    
    @Override
    public List<String> splitTokens(String input) {
        if (normalizer != null) {
            if (normalizer.isLowercase()) input = input.toLowerCase();
        }
        
        List<String> output = new ArrayList<>();
        
        for (String word : input.split("(?=\\p{Punct})|(?<=\\p{Punct})|\\s+")) {
            word = word.replaceAll(" ", "");
            output.addAll(encodeWordPiece(word));
        }
        
        return output;
    }
    
    @Override
    public void load(File file) throws IOException {
        if (!file.exists()) throw new FileNotFoundException(file.getPath());
        
        try (Reader reader = new FileReader(file)) {
            JsonObject root = GSON.fromJson(reader, JsonObject.class);
            JsonObject model = root.getAsJsonObject("model");
            
            this.unkToken = model.get("unk_token").getAsString();
            
            if (root.has("normalizer") && root.get("normalizer").isJsonObject()) {
                this.normalizer = GSON.fromJson(root.getAsJsonObject("normalizer"), Normalizer.class);
            }
            
            Type tokenListType = new TypeToken<List<AddedToken>>() {}.getType();
            this.addedTokens = GSON.fromJson(root.getAsJsonArray("added_tokens"), tokenListType);
            
            Type vocabType = new TypeToken<LinkedHashMap<String, Integer>>() {}.getType();
            this.vocab = GSON.fromJson(model.getAsJsonObject("vocab"), vocabType);
        }
    }
    
    @Override
    public void save(File file) throws IOException {
        if (!file.exists() && !file.getParentFile().mkdirs()) {
            throw new IOException("Cannot create directory: " + file);
        }
        
        JsonObject root = new JsonObject();
        root.addProperty("version", "1.0");
        root.add("truncation", null);
        root.add("padding", null);
        root.add("added_tokens", GSON.toJsonTree(addedTokens));
        root.add("normalizer", normalizer == null ? null : GSON.toJsonTree(normalizer));
        
        JsonObject preTokenizer = new JsonObject();
        preTokenizer.addProperty("type", "BertPreTokenizer");
        root.add("pre_tokenizer", preTokenizer);
        
        JsonObject decoder = new JsonObject();
        decoder.addProperty("type", "WordPiece");
        decoder.addProperty("prefix", tokenStarter);
        decoder.addProperty("cleanup", true);
        root.add("decoder", decoder);
        
        JsonObject model = new JsonObject();
        model.add("vocab", GSON.toJsonTree(vocab));
        root.add("model", model);
        
        try (Writer writer = new FileWriter(file)) {
            GSON.toJson(root, writer);
        }
    }
    
    public List<String> encodeWordPiece(String word) {
        List<String> tokens = new ArrayList<>();
        int start = 0;
        
        while (start < word.length()) {
            int end = word.length();
            String curSubstr = null;
            
            while (start < end) {
                String substr = (start == 0) ? word.substring(start, end)
                    : "##" + word.substring(start, end);
                if (vocab.containsKey(substr)) {
                    curSubstr = substr;
                    break;
                }
                end -= 1;
            }
            
            if (curSubstr == null) {
                tokens.add("[UNK]");
                break;
            }
            
            tokens.add(curSubstr);
            start = end;
        }
        
        return tokens;
    }
}
