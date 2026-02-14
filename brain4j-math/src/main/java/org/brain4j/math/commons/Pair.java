package org.brain4j.math.commons;

public class Pair<K, V> {

    protected K first;
    protected V second;

    public Pair(K key, V second) {
        this.first = key;
        this.second = second;
    }

    public K getFirst() {
        return first;
    }

    public void setFirst(K first) {
        this.first = first;
    }

    public V getSecond() {
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
