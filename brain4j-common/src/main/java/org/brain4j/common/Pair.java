package org.brain4j.common;

public class Pair<K, V> {

    private K first;
    private V second;

    public Pair(K key, V second) {
        this.first = key;
        this.second = second;
    }

    public K first() {
        return first;
    }

    public void setFirst(K first) {
        this.first = first;
    }

    public V second() {
        return second;
    }

    public void setSecond(V second) {
        this.second = second;
    }

    @Override
    public String toString() {
        return first + " -> " + second;
    }
}
