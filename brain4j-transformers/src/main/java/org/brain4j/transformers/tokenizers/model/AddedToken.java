package org.brain4j.transformers.tokenizers.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public record AddedToken(
        int id,
        String content,
        @JsonProperty("single_word") boolean singleWord,
        boolean lstrip,
        boolean rstrip,
        boolean normalized,
        boolean special
) {}
