package org.brain4j.core.transformer.impl;

import org.brain4j.core.transformer.Tokenizer;

import java.util.*;

public class BytePairTokenizer implements Tokenizer {

    private final int numMerges;
    private final Map<String, Integer> encodings;

    public BytePairTokenizer(int numMerges) {
        this.numMerges = numMerges;
        this.encodings = new LinkedHashMap<>();
    }

    @Override
    public List<String> tokenize(String input) {
        List<String> output = new ArrayList<>();

        for (String word : input.split("\\s+")) {
            output.addAll(encodeWord(word));
        }

        return output;
    }

    public void fit(List<String> corpus) {
        Map<String, String[]> tokens = new LinkedHashMap<>();

        for (String word : corpus) {
            String token = String.join(" ", word.split(""));
            tokens.put(token, token.split("\\s+"));
        }

        for (int iter = 0; iter < numMerges; iter++) {
            Map<String, Integer> pairCounts = new HashMap<>();

            for (String[] symbols : tokens.values()) {
                for (int i = 0; i < symbols.length - 1; i++) {
                    String pair = symbols[i] + symbols[i + 1];
                    pairCounts.merge(pair, 1, Integer::sum);
                }
            }

            if (pairCounts.isEmpty()) break;

            String bestPair = Collections.max(pairCounts.entrySet(), Map.Entry.comparingByValue()).getKey();
            encodings.put(bestPair, iter);

            Map<String, String[]> updated = new LinkedHashMap<>();

            for (Map.Entry<String, String[]> entry : tokens.entrySet()) {
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

            tokens = updated;
        }
    }

    public List<String> encodeWord(String word) {
        List<String> symbols = new ArrayList<>(Arrays.asList(word.split("")));
        symbols.add("</w>");

        while (true) {
            Map<String, Integer> candidates = new HashMap<>();

            for (int i = 0; i < symbols.size() - 1; i++) {
                String pair = symbols.get(i) + symbols.get(i + 1);
                int rank = encodings.getOrDefault(pair, Integer.MAX_VALUE);

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

    public Map<String, Integer> encodings() {
        return Collections.unmodifiableMap(encodings);
    }
}
