package org.brain4j.transformers.api;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ModelInfo(
    @JsonProperty("id") String id,
    @JsonProperty("sha") String sha,
    @JsonProperty("author") String author,
    @JsonProperty("private") boolean isPrivate,
    @JsonProperty("tags") List<String> tags,
    @JsonProperty("lastModified") String lastModified,
    @JsonProperty("downloads") long downloads,
    @JsonProperty("pipeline_tag") String pipelineTag,
    @JsonProperty("siblings") List<Sibling> siblings
) {
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Sibling(
        @JsonProperty("rfilename") String rfilename,
        @JsonProperty("size") long size,
        @JsonProperty("lfs") Object lfs
    ) { }
}