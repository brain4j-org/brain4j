package net.echo.brain4j.transformers;

import net.echo.brain4j.structure.Token;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class Vocabulary {

    private final Map<String, Token> tokenMap;
    private final Map<Integer, Token> idMap;
    private final Map<String, Tensor> tensorCache;

    public Vocabulary(List<String> corpus) {
        this.tokenMap = new HashMap<>();
        this.idMap = new HashMap<>();
        this.tensorCache = new ConcurrentHashMap<>();

        Set<String> distinctTokens = new HashSet<>(corpus);
        int index = 0;

        for (String token : distinctTokens) {
            Token newToken = new Token(token, index);

            tokenMap.put(token, newToken);
            idMap.put(index, newToken);

            index++;
        }
    }

    public int size() {
        return tokenMap.size();
    }

    public int getId(String token) {
        return tokenMap.getOrDefault(token, new Token("<unk>", 0)).id();
    }

    public String getToken(int id) {
        return idMap.getOrDefault(id, new Token("<unk>", 0)).text();
    }

    public Tensor getTensor(String token, int embeddingDim) {
        return tensorCache.computeIfAbsent(token, t -> {
            int id = getId(t);

            Tensor result = TensorFactory.zeros(embeddingDim);
            Random random = Random.from(new SplittableRandom(id));

            for (int i = 0; i < embeddingDim; i++) {
                result.set(random.nextFloat() * 2 - 1, i);
            }

            return result;
        });
    }
}
