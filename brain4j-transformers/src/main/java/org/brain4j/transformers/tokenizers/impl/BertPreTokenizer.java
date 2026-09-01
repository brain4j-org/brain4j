package org.brain4j.transformers.tokenizers.impl;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.transformers.tokenizers.model.AddedToken;
import org.brain4j.transformers.tokenizers.model.Normalizer;

import java.io.*;
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
            if (normalizer.lowercase()) input = input.toLowerCase();
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
            JsonNode root = MAPPER.readTree(reader);
            JsonNode model = root.get("model");
            
            if (model != null && model.has("unk_token")) {
                JsonNode unkNode = model.get("unk_token");
                if (unkNode != null && unkNode.isTextual()) {
                    this.unkToken = unkNode.asText();
                }
            }
            
            if (root.has("normalizer") && root.get("normalizer").isObject()) {
                this.normalizer = MAPPER.treeToValue(root.get("normalizer"), Normalizer.class);
            }
            
            JsonNode addedTokensNode = root.get("added_tokens");
            if (addedTokensNode != null && !addedTokensNode.isNull()) {
                this.addedTokens = MAPPER.convertValue(addedTokensNode, new TypeReference<List<AddedToken>>() {});
            } else {
                this.addedTokens = new ArrayList<>();
            }
            
            JsonNode vocabNode = model != null ? model.get("vocab") : null;
            if (vocabNode != null && !vocabNode.isNull()) {
                this.vocab = MAPPER.convertValue(vocabNode, new TypeReference<LinkedHashMap<String, Integer>>() {});
            }
        }
    }
    
    @Override
    public void save(File file) throws IOException {
        if (!file.exists() && !file.getParentFile().mkdirs()) {
            throw new IOException("Cannot create directory: " + file);
        }
        
        ObjectNode root = MAPPER.createObjectNode();
        root.put("version", "1.0");
        root.set("truncation", MAPPER.nullNode());
        root.set("padding", MAPPER.nullNode());
        root.set("added_tokens", MAPPER.valueToTree(addedTokens));
        root.set("normalizer", normalizer == null ? MAPPER.nullNode() : MAPPER.valueToTree(normalizer));
        
        ObjectNode preTokenizer = MAPPER.createObjectNode();
        preTokenizer.put("type", "BertPreTokenizer");
        root.set("pre_tokenizer", preTokenizer);
        
        ObjectNode decoder = MAPPER.createObjectNode();
        decoder.put("type", "WordPiece");
        decoder.put("prefix", tokenStarter);
        decoder.put("cleanup", true);
        root.set("decoder", decoder);
        
        ObjectNode model = MAPPER.createObjectNode();
        model.set("vocab", MAPPER.valueToTree(vocab));
        root.set("model", model);
        
        try (Writer writer = new FileWriter(file)) {
            MAPPER.writerWithDefaultPrettyPrinter().writeValue(writer, root);
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
