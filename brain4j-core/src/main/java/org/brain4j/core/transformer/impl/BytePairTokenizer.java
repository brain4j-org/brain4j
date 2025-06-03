package org.brain4j.core.transformer.impl;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.brain4j.core.transformer.Tokenizer;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;

public class BytePairTokenizer implements Tokenizer {

    public static class TokenizerState {
        Map<String, Integer> encodings = new LinkedHashMap<>();
        Map<String, String[]> tokens = new LinkedHashMap<>();
    }

    private final ForkJoinPool threadPool = ForkJoinPool.commonPool();
    private final int numMerges;
    private TokenizerState state;

    public BytePairTokenizer(int numMerges) {
        this.numMerges = numMerges;
        this.state = new TokenizerState();
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
    public void save(String path) throws IOException {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();

        try (Writer writer = new FileWriter(path)) {
            gson.toJson(state, writer);
        }
    }

    @Override
    public void load(String path) throws IOException {
        Gson gson = new Gson();

        try (Reader reader = new FileReader(path)) {
            this.state = gson.fromJson(reader, TokenizerState.class);
        }
    }

    public void fit(List<String> corpus) {
        int totalTokens = 0;

        if (state.tokens.isEmpty()) {
            for (String word : corpus) {
                String token = String.join(" ", word.split(""));
                String[] symbols = token.split("\\s+");

                state.tokens.put(token, symbols);
            }
        }

        for (String[] symbols : state.tokens.values()) {
            totalTokens += symbols.length;
        }

        System.out.println("Total of " + totalTokens + " tokens");

        for (int iter = 0; iter < numMerges; iter++) {
            long start = System.nanoTime();

            Map<String, Integer> pairCounts = new HashMap<>();
            List<Callable<Void>> tasks = new ArrayList<>();

            for (String[] symbols : state.tokens.values()) {
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
            state.encodings.put(bestPair, iter);

            Map<String, String[]> updated = new LinkedHashMap<>();

            for (Map.Entry<String, String[]> entry : state.tokens.entrySet()) {
                String[] symbols = entry.getValue();
                List<String> merged = new ArrayList<>();

                for (int i = 0; i < symbols.length;) {
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

            state.tokens.clear();
            state.tokens.putAll(updated);

            double took = (System.nanoTime() - start) / 1e6;

            System.out.println("Completed merge " + (iter + 1) + " in " + took + " ms");
            System.gc();
        }
    }

    public List<String> encodeWord(String word) {
        List<String> symbols = new ArrayList<>(Arrays.asList(word.split("")));
        symbols.add("</w>");

        while (true) {
            Map<String, Integer> candidates = new HashMap<>();

            for (int i = 0; i < symbols.size() - 1; i++) {
                String pair = symbols.get(i) + symbols.get(i + 1);
                int rank = state.encodings.getOrDefault(pair, Integer.MAX_VALUE);

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

    public TokenizerState state() {
        return state;
    }

    public Map<String, Integer> encodings() {
        return Collections.unmodifiableMap(state.encodings);
    }

    public Map<String, String[]> tokens() {
        return Collections.unmodifiableMap(state.tokens);
    }

    public void clearTokens() {
        state.tokens.clear();
    }

    public void clearEncodings() {
        state.encodings.clear();
    }
}
