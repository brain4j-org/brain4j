package org.brain4j.math.commons;

/**
 * Represents a numeric range used for tensor slicing and indexing operations.
 * <p>
 * A {@code Range} defines a sequence of indices in the form:
 * <pre>{@code
 * start : end : step
 * }</pre>
 * Similar to Python-style slicing semantics, it supports both positive and negative
 * indices, where negative values count from the end of the dimension.
 * <p>
 * Examples:
 * <ul>
 *     <li>{@code Range.interval(0, 5)} selects elements [0, 1, 2, 3, 4]</li>
 *     <li>{@code Range.point(3)} selects only element [3]</li>
 *     <li>{@code new Range(-3, -1)} selects the third-to-last and second-to-last elements</li>
 * </ul>
 *
 * @author xEcho1337
 * @since 3.0
 */
public record Range(int start, int end, int step) {
    /**
     * Creates a new range covering the interval [start, end) with a default step of 1.
     *
     * @param start the starting index (inclusive)
     * @param end   the ending index (exclusive)
     * @return a new {@code Range} object
     */
    public static Range interval(int start, int end) {
        return new Range(start, end);
    }

    /**
     * Creates a range that spans all possible indices.
     * <p>
     * This is typically used to represent a full-dimension selection,
     * equivalent to {@code ":"} in NumPy-like syntax.
     *
     * @return a {@code Range} from 0 to {@link Integer#MAX_VALUE} with step 1
     */
    public static Range all() {
        return new Range(0, Integer.MAX_VALUE, 1);
    }

    /**
     * Creates a range representing a single point (a single index).
     *
     * @param i the index to select
     * @return a {@code Range} selecting only element {@code i}
     */
    public static Range point(int i) {
        return new Range(i, i + 1, 1);
    }

    /**
     * Constructs a {@code Range} from {@code start} to {@code end} with a step of 1.
     *
     * @param start the starting index (inclusive)
     * @param end   the ending index (exclusive)
     */
    public Range(int start, int end) {
        this(start, end, 1);
    }

    /**
     * Constructs a {@code Range} with the specified start, end, and step values.
     *
     * @param start the starting index (inclusive)
     * @param end   the ending index (exclusive)
     * @param step  the increment between consecutive indices (must be non-zero)
     */
    public Range {
    }

    /**
     * Computes the normalized start index for a given dimension size.
     * <p>
     * If {@code start} is negative, it is offset from the end of the dimension.
     *
     * @param dimSize the size of the dimension being indexed
     * @return the resolved (non-negative) start index
     */
    public int start(int dimSize) {
        return start >= 0 ? start : start + dimSize;
    }

    /**
     * Computes the normalized end index for a given dimension size.
     * <p>
     * If {@code end} is negative, it is offset from the end of the dimension.
     * The result is clamped to the range [0, dimSize].
     *
     * @param dimSize the size of the dimension being indexed
     * @return the resolved (non-negative) end index
     */
    public int end(int dimSize) {
        return (end >= 0 && start >= 0) ? Math.min(end, dimSize) : end + dimSize;
    }

    /**
     * Returns the number of elements covered by this range for a given dimension.
     *
     * @param dimSize the size of the dimension being indexed
     * @return the computed number of elements
     */
    public int size(int dimSize) {
        int s = start(dimSize);
        int e = end(dimSize);
        return (e - s + step - 1) / step;
    }

    /**
     * Returns the raw (possibly negative) start index.
     *
     * @return the start index
     */
    @Override
    public int start() {
        return start;
    }

    /**
     * Returns the raw (possibly negative) end index.
     *
     * @return the end index
     */
    @Override
    public int end() {
        return end;
    }

    /**
     * Returns the step value used to iterate through the range.
     *
     * @return the step size
     */
    @Override
    public int step() {
        return step;
    }

    /**
     * Returns a string representation of the range in the format {@code "start:end:step"}.
     *
     * @return a human-readable representation of this range
     */
    @Override
    public String toString() {
        return start + ":" + end + ":" + step;
    }
}