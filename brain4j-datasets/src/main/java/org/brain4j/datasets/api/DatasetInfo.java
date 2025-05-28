package org.brain4j.datasets.api;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public record DatasetInfo(
        @JsonProperty("_id") String internalId,
        @JsonProperty("id") String id,
        @JsonProperty("author") String author,
        @JsonProperty("sha") String sha,
        @JsonProperty("lastModified") String lastModified,
        @JsonProperty("private") boolean isPrivate,
        @JsonProperty("gated") boolean gated,
        @JsonProperty("disabled") boolean disabled,
        @JsonProperty("tags") List<String> tags,
        @JsonProperty("citation") String citation,
        @JsonProperty("description") String description,
        @JsonProperty("paperswithcode_id") String paperswithcodeId,
        @JsonProperty("downloads") long downloads,
        @JsonProperty("likes") long likes,
        @JsonProperty("cardData") Map<String, Object> cardData,
        @JsonProperty("siblings") List<DatasetFile> siblings,
        @JsonProperty("createdAt") String createdAt,
        @JsonProperty("usedStorage") long usedStorage
) {

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record DatasetFile(
            @JsonProperty("rfilename") String filename
    ) {}
}