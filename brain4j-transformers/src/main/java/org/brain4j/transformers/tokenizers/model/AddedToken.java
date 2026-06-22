package org.brain4j.transformers.tokenizers.model;

public class AddedToken {
    
    private int id;
    private String content;
    private boolean single_word;
    private boolean lstrip;
    private boolean rstrip;
    private boolean normalized;
    private boolean special;
    
    public int getId() {
        return id;
    }
    
    public String getContent() {
        return content;
    }
    
    public boolean isSingle_word() {
        return single_word;
    }
    
    public boolean isLstrip() {
        return lstrip;
    }
    
    public boolean isRstrip() {
        return rstrip;
    }
    
    public boolean isNormalized() {
        return normalized;
    }
    
    public boolean isSpecial() {
        return special;
    }
}
