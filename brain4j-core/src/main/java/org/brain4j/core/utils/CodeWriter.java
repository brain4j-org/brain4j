package org.brain4j.core.utils;

public class CodeWriter {
    
    private int indentation = 0;
    private String buffer;
    
    public CodeWriter() {
        this.buffer = "";
    }
    
    public void indent() {
        indentation++;
    }
    
    public void unindent() {
        indentation--;
    }
    
    public void write(String text) {
        buffer += " ".repeat(indentation * 4) + text;
    }
    
    public void writeLine(String line) {
        write(line + "\n");
    }
    
    @Override
    public String toString() {
        return buffer;
    }
}
