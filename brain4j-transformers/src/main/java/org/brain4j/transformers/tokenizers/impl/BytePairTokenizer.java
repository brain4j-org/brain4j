package org.brain4j.transformers.tokenizers.impl;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.brain4j.core.utils.Colored;
import org.brain4j.transformers.tokenizers.model.AddedToken;
import org.brain4j.transformers.tokenizers.model.Normalizer;
import org.brain4j.transformers.tokenizers.model.Tokenizer;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Tensor;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

/**
 * Implementation of Byte Pair Encoding (BPE) tokenization algorithm.
 *
 * <p>BPE is a data compression technique that iteratively replaces the most frequent
 * pairs of bytes (or characters) in a sequence with a single, unused byte. In NLP,
 * it's used to build vocabulary by merging frequent character pairs into new tokens.
 *
 * <p>This tokenizer supports:
 * <ul>
 *   <li>Vocabulary loading/saving
 *   <li>Token splitting with optional word prefix
 *   <li>Special token handling (BOS/EOS)
 * </ul>
 *
 * <p>Usage example:
 * <pre>{@code
 * BytePairTokenizer tokenizer = new BytePairTokenizer("Ġ");  // GPT-style prefix
 * tokenizer.loadVocab("vocab.json");
 * List<String> tokens = tokenizer.splitTokens("Hello world");
 * }</pre>
 */
public class BytePairTokenizer implements Tokenizer {
    
    public static final ObjectMapper MAPPER = new ObjectMapper();
    
    protected Normalizer normalizer;
    protected List<AddedToken> addedTokens;
    protected Map<String, Integer> vocab;
    protected Map<String, String[]> merges;
    protected final String tokenStarter;
    protected String unkToken;
    protected int bosTokenId;
    protected int eosTokenId;
    
    public BytePairTokenizer() {
        this("Ġ");
    }
    
    public BytePairTokenizer(String tokenStarter) {
        this.addedTokens = new ArrayList<>();
        this.vocab = new LinkedHashMap<>();
        this.merges = new LinkedHashMap<>();
        this.tokenStarter = tokenStarter;
        this.unkToken = "[UNK]";
    }
    
    @Override
    public List<String> splitTokens(String input) {
        if (normalizer != null) {
            if (normalizer.lowercase()) input = input.toLowerCase();
        }
        
        List<String> output = new ArrayList<>();

        for (String word : input.split("\\s+")) {
            boolean hasCharBeforeWord = input.indexOf(word) != 0;
            if (hasCharBeforeWord) word = tokenStarter + word;
            output.addAll(encodeWord(word));
        }

        return output;
    }

    @Override
    public String decode(int index) {
        Optional<Map.Entry<String, Integer>> token = vocab.entrySet()
                .stream()
                .filter(x -> x.getValue() == index)
                .findFirst();

        return token.map(entry -> entry.getKey().replace(tokenStarter, " ")).orElse("<|unk|>");

    }

    @Override
    public Tensor encode(List<String> tokens) {
        Tensor result = Tensors.zeros(tokens.size());

        for (int i = 0; i < tokens.size(); i++) {
            int index = vocab.get(tokens.get(i));
            result.set(index, i);
        }
        
        return result;
    }
    
    @Override
    public int vocabSize() {
        return vocab.size();
    }
    
    @Override
    public int bosTokenId() {
        return bosTokenId;
    }
    
    @Override
    public void setBosTokenId(int bosTokenId) {
        this.bosTokenId = bosTokenId;
    }
    
    @Override
    public int eosTokenId() {
        return eosTokenId;
    }
    
    @Override
    public void setEosTokenId(int eosTokenId) {
        this.eosTokenId = eosTokenId;
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
        preTokenizer.put("type", "ByteLevel");
        preTokenizer.put("add_prefix_space", false);
        root.set("pre_tokenizer", preTokenizer);
        
        ObjectNode postProcessor = MAPPER.createObjectNode();
        postProcessor.put("type", "ByteLevel");
        postProcessor.put("trim_offsets", true);
        root.set("post_processor", postProcessor);
        
        ObjectNode decoder = MAPPER.createObjectNode();
        decoder.put("type", "ByteLevel");
        decoder.put("add_prefix_space", false);
        root.set("decoder", decoder);
        
        ObjectNode model = MAPPER.createObjectNode();
        model.put("type", "BPE");
        model.set("vocab", MAPPER.valueToTree(vocab)); // Map<String, Integer>
        
        List<String> mergeStrings = new ArrayList<>();
        
        for (String[] pair : merges.values()) {
            if (pair.length != 2) continue;
            
            mergeStrings.add(pair[0] + " " + pair[1]);
        }
        
        model.put("unk_token", unkToken);
        model.set("merges", MAPPER.valueToTree(mergeStrings));
        model.set("dropout", MAPPER.nullNode());
        root.set("model", model);
        
        try (Writer writer = new FileWriter(file)) {
            MAPPER.writerWithDefaultPrettyPrinter().writeValue(writer, root);
        }
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
                this.addedTokens = MAPPER.convertValue(addedTokensNode, new TypeReference<>() {
                });
            } else {
                this.addedTokens = new ArrayList<>();
            }
            
            JsonNode vocabNode = model != null ? model.get("vocab") : null;
            if (vocabNode != null && !vocabNode.isNull()) {
                this.vocab = MAPPER.convertValue(vocabNode, new TypeReference<LinkedHashMap<String, Integer>>() {});
            }
            
            JsonNode mergesNode = model != null ? model.get("merges") : null;
            List<String> mergeStrings = Collections.emptyList();
            if (mergesNode != null && mergesNode.isArray()) {
                mergeStrings = MAPPER.convertValue(mergesNode, new TypeReference<>() {
                });
            }
            
            LinkedHashMap<String, String[]> loaded = new LinkedHashMap<>();
            
            for (String merge : mergeStrings) {
                String[] pair = merge.split(" ");
                
                if (pair.length == 2) {
                    loaded.put(pair[0] + pair[1], pair);
                }
            }
            
            this.merges = loaded;
        }
    }

    public void fit(List<String> corpus, int numMerges, int evaluateDelay) {
        if (merges.isEmpty()) {
            for (String word : corpus) {
                String token = String.join(" ", word.split(""));
                String[] symbols = token.split("\\s+");

                merges.put(token, symbols);
            }
        }

        for (int iter = 0; iter < numMerges; iter++) {
            long start = System.nanoTime();

            Map<String, Integer> pairCounts = new ConcurrentHashMap<>();
            List<RecursiveAction> tasks = new ArrayList<>();

            for (String[] symbols : merges.values()) {
                RecursiveAction task = new RecursiveAction() {
                    @Override
                    protected void compute() {
                        for (int i = 0; i < symbols.length - 1; i++) {
                            String pair = symbols[i] + symbols[i + 1];
                            pairCounts.merge(pair, 1, Integer::sum);
                        }
                    }
                };

                tasks.add(task);
            }
            
            ForkJoinPool.commonPool().submit(() -> ForkJoinTask.invokeAll(tasks)).join();

            if (pairCounts.isEmpty()) break;

            String bestPair = Collections.max(pairCounts.entrySet(), Map.Entry.comparingByValue()).getKey();
            vocab.put(bestPair, iter);

            Map<String, String[]> updated = new LinkedHashMap<>();

            for (Map.Entry<String, String[]> entry : merges.entrySet()) {
                String[] symbols = entry.getValue();
                List<String> merged = new ArrayList<>();

                for (int i = 0; i < symbols.length; ) {
                    if (i < symbols.length - 1 && (symbols[i] + symbols[i + 1]).equals(bestPair)) {
                        merged.add(bestPair);
                        i += 2;
                    } else {
                        merged.add(symbols[i]);
                        i++;
                    }
                }

                String key = String.join(" ", merged);
                updated.put(key, merged.toArray(new String[0]));
            }

            merges.clear();
            merges.putAll(updated);

            double took = (System.nanoTime() - start) / 1e6;
            printProgressBar(took, iter, numMerges, evaluateDelay);
        }
    }

    public List<String> encodeWord(String word) {
        List<String> symbols = new ArrayList<>(Arrays.asList(word.split("")));
        symbols.add("</w>");

        while (true) {
            Map<String, Integer> candidates = new HashMap<>();

            for (int i = 0; i < symbols.size() - 1; i++) {
                String pair = symbols.get(i) + symbols.get(i + 1);
                int rank = vocab.getOrDefault(pair, Integer.MAX_VALUE);

                candidates.put(pair, rank);
            }

            String best = null;
            int bestRank = Integer.MAX_VALUE;

            for (Map.Entry<String, Integer> entry : candidates.entrySet()) {
                if (entry.getValue() < bestRank) {
                    best = entry.getKey();
                    bestRank = entry.getValue();
                }
            }

            if (best == null) break;

            symbols = getSymbols(symbols, best);
        }

        if (!symbols.isEmpty() && symbols.getLast().equals("</w>")) {
            symbols.removeLast();
        }

        return symbols;
    }

    private static List<String> getSymbols(List<String> symbols, String best) {
        List<String> newSymbols = new ArrayList<>();

        for (int i = 0; i < symbols.size(); ) {
            String cur = symbols.get(i);
            String next = (i < symbols.size() - 1) ? symbols.get(i + 1) : null;

            if (next != null && (cur + next).equals(best)) {
                newSymbols.add(best);
                i += 2;
            } else {
                newSymbols.add(cur);
                i++;
            }
        }

        return newSymbols;
    }

    private void printProgressBar(
        double tookMs,
        int iteration,
        int merges,
        int evaluateDelay
    ) {
        int progressBarLength = 20;
        double percentage = (double) iteration / merges;

        String barChar = Commons.HEADER_CHAR;
        double seconds = tookMs / 1000;

        String timeStr = Commons.formatDuration(seconds);
        String progressBar = Commons.createProgressBar(
            percentage, progressBarLength,
            "<green>", barChar,
            "<reset>", barChar
        );
        String format = "<reset>[%s/%s] %s <gray>[%s/iter]<reset>";
        String message = String.format(format, iteration, merges, progressBar, timeStr);

        System.out.print(Colored.renderText(message));

        if (iteration % evaluateDelay == 0) {
            printEvaluation(iteration, merges);
        }
    }

    private void printEvaluation(int iteration, int total) {
        System.out.println();

        String symbolsMsg = "Symbols: <blue>%,d<reset>";
        String tokensMsg = "Tokens: <green>%,d<reset>";

        String message = "[%s/%s] " + symbolsMsg + " | " + tokensMsg + "\n";
        String formatted = String.format(message, iteration, total, merges.size(), totalSymbols());

        System.out.print(formatted);
    }

    private int totalSymbols() {
        return merges.values()
                .stream()
                .mapToInt(x -> x.length)
                .sum();
    }
    
    public Normalizer getNormalizer() {
        return normalizer;
    }
    
    public List<AddedToken> getAddedTokens() {
        return addedTokens;
    }
    
    public Map<String, Integer> getVocab() {
        return Collections.unmodifiableMap(vocab);
    }
    
    public Map<String, String[]> getMerges() {
        return Collections.unmodifiableMap(merges);
    }

    public void clearTokens() {
        merges.clear();
    }

    public void clearEncodings() {
        vocab.clear();
    }
}
