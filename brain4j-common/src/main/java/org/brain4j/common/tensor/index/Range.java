package org.brain4j.common.tensor.index;

public class Range {
    private final int start;
    private final int end;
    private final int step;

    public static Range all() {
        return new Range(0, Integer.MAX_VALUE);
    }

    public Range(int start, int end) {
        this(start, end, 1);
    }
    
    public Range(int start, int end, int step) {
        this.start = start;
        this.end = end;
        this.step = step;
    }
    
    public int start(int dimSize) {
        return start >= 0 ? start : start + dimSize;
    }
    
    public int end(int dimSize) {
        return end >= 0 ? Math.min(end, dimSize) : end + dimSize;
    }
    
    public int step() {
        return step;
    }
    
    public int size(int dimSize) {
        int s = start(dimSize);
        int e = end(dimSize);
        return (e - s + step - 1) / step;
    }
} 