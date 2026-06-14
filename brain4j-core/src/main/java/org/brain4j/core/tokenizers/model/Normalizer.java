package org.brain4j.core.tokenizers.model;

import com.google.gson.annotations.SerializedName;

public class Normalizer {
    
    @SerializedName("type")
    private String type;
    @SerializedName("clean_text")
    private boolean cleanText;
    @SerializedName("handle_chinese_chars")
    private boolean handleChineseChars;
    @SerializedName("lowercase")
    private boolean lowercase;
    
    public String getType() {
        return type;
    }
    
    public boolean isCleanText() {
        return cleanText;
    }
    
    public boolean isHandleChineseChars() {
        return handleChineseChars;
    }
    
    public boolean isLowercase() {
        return lowercase;
    }
}
