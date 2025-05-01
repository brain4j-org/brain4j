package org.brain4j.core.transformers;

import org.brain4j.core.structure.Token;
import org.brain4j.core.transformers.tokenizer.Tokenizer;

import java.util.*;

public class Vocabulary {

    private final Map<String, Token> tokenMap;
    private final Map<Integer, Token> idMap;
    private int globalIndex;

    public Vocabulary(Tokenizer tokenizer, List<String> corpus) {
        this.tokenMap = new HashMap<>();
        this.idMap = new HashMap<>();

        for (String text : corpus) {
            List<String> allTokens = tokenizer.split(text);
            Set<String> distinctTokens = new HashSet<>(allTokens);

            for (String token : distinctTokens) {
                Token newToken = new Token(token, globalIndex);

                tokenMap.put(token, newToken);
                idMap.put(globalIndex, newToken);

                globalIndex++;
            }
        }
    }

    public int size() {
        return globalIndex;
    }

    public int getId(String token) {
        return tokenMap.getOrDefault(token, new Token("<unk>", 0)).id();
    }

    public String getToken(int id) {
        return idMap.getOrDefault(id, new Token("<unk>", 0)).text();
    }
}
