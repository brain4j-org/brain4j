package org.brain4j.core.transformer.tokenizers.impl;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import org.brain4j.core.transformer.tokenizers.Tokenizer;
import org.brain4j.math.Commons;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.*;

import static org.brain4j.math.constants.Constants.*;
import static org.brain4j.math.constants.Constants.RESET;

public class BytePairTokenizer implements Tokenizer {

    private static final Logger logger = LoggerFactory.getLogger(BytePairTokenizer.class);
    private static final Logger trainLogger = LoggerFactory.getLogger("dynamic");

    private final ForkJoinPool threadPool = ForkJoinPool.commonPool();

    private Map<String, Integer> vocab;
    private Map<String, String[]> merges;
    
    public BytePairTokenizer() {
        this.vocab = new LinkedHashMap<>();
        this.merges = new LinkedHashMap<>();
    }

    @Override
    public List<String> tokenize(String input) {
        List<String> output = new ArrayList<>();

        for (String word : input.split("\\s+")) {
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

        if (token.isEmpty()) {
            return "<|unk|>";
        }

        return token.get().getKey();
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
    public void save(String filePath) throws IOException {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();

        File file = new File(filePath);

        if (!file.exists()) {
            boolean result = file.mkdir();

            if (!result) {
                throw new IllegalStateException("Failed to create directory.");
            }
        }

        String vocabFile = filePath + "/vocab.json";
        String mergesFile = filePath + "/merges.txt";

        try (Writer writer = new FileWriter(vocabFile)) {
            gson.toJson(vocab, writer);
        }

        try (Writer writer = new FileWriter(mergesFile)) {
            StringBuilder result = new StringBuilder();

            merges.forEach((key, value) -> result
                    .append(key)
                    .append(" ")
                    .append(String.join("", value))
                    .append("\n"));

            writer.write(result.toString());
            writer.flush();
        }
    }

    @Override
    public void load(String path) throws IOException {
        Gson gson = new Gson();
        File dir = new File(path);

        if (!dir.exists() || !dir.isDirectory()) {
            throw new FileNotFoundException("Directory does not exist: " + path);
        }

        File[] files = dir.listFiles();

        if (files == null) {
            throw new IOException("Failed to list files in directory: " + path);
        }

        for (File child : files) {
            if (child.getName().equals("vocab.json")) {
                try (Reader reader = new FileReader(child)) {
                    Type type = new TypeToken<LinkedHashMap<String, Integer>>() {}.getType();
                    this.vocab = gson.fromJson(reader, type);
                }
            }

            if (child.getName().equals("merges.txt")) {
                Map<String, String[]> merges = new LinkedHashMap<>();
                List<String> lines = Files.readAllLines(child.toPath());

                for (String line : lines) {
                    line = line.trim();

                    if (line.isEmpty() || line.startsWith("#")) continue;

                    String[] pair = line.split("\\s+");

                    if (pair.length == 2) {
                        String key = pair[0] + pair[1];
                        merges.put(key, pair);
                    }
                }

                this.merges = merges;
            }
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

        int totalSymbols = totalSymbols();

        logger.debug("Total symbols: {}", totalSymbols);
        logger.debug("Total tokens: {}", merges.size());

        for (int iter = 0; iter < numMerges; iter++) {
            long start = System.nanoTime();

            Map<String, Integer> pairCounts = new ConcurrentHashMap<>();
            List<Callable<Void>> tasks = new ArrayList<>();

            for (String[] symbols : merges.values()) {
                Callable<Void> task = () -> {
                    for (int i = 0; i < symbols.length - 1; i++) {
                        String pair = symbols[i] + symbols[i + 1];
                        pairCounts.merge(pair, 1, Integer::sum);
                    }
                    return null;
                };

                tasks.add(task);
            }

            threadPool.invokeAll(tasks);

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

        String barChar = Commons.getHeaderChar();
        int remaining = merges - iteration;

        double seconds = tookMs / 1000;
        double remainingTime = seconds * remaining;

        String remainingTimeStr = Commons.formatDuration(remainingTime);
        String timeStr = Commons.formatDuration(seconds);

        String progressMsg = WHITE + "[%s/%s] ";
        String progressBar = LIGHT_GREEN + Commons.createProgressBar(
                percentage,
                progressBarLength,
                barChar,
                RESET + barChar
        );

        String percentual = LIGHT_YELLOW + " %.2f%%" + RESET;
        String time = GRAY + " [%s/epoch | %s remaining]" + RESET;
        String message = String.format(progressMsg + progressBar + percentual + time,
                iteration, merges, percentage * 100, timeStr, remainingTimeStr);

        trainLogger.info(message);

        if (iteration % evaluateDelay == 0) {
            printEvaluation(iteration, merges);
        }
    }

    private void printEvaluation(int iteration, int total) {
        System.out.println();

        String symbolsMsg = "Symbols: " + LIGHT_BLUE + "%,d" + RESET;
        String tokensMsg = "Tokens: " + LIGHT_GREEN + "%,d" + RESET;

        String message = "[%s/%s] " + symbolsMsg + " | " + tokensMsg + "\n";
        String formatted = String.format(message, iteration, total, merges.size(), totalSymbols());

        trainLogger.info(formatted);
    }

    private int totalSymbols() {
        return merges.values()
                .stream()
                .mapToInt(x -> x.length)
                .sum();
    }

    public Map<String, Integer> vocab() {
        return Collections.unmodifiableMap(vocab);
    }

    public Map<String, String[]> merges() {
        return Collections.unmodifiableMap(merges);
    }

    public void clearTokens() {
        merges.clear();
    }

    public void clearEncodings() {
        vocab.clear();
    }
}
