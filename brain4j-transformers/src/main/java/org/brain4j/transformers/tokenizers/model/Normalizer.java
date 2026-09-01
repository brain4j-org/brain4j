package org.brain4j.transformers.tokenizers.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public record Normalizer(
        @JsonProperty("type") String type,
        @JsonProperty("clean_text") boolean cleanText,
        @JsonProperty("handle_chinese_chars") boolean handleChineseChars,
        @JsonProperty("lowercase") boolean lowercase
) {}
