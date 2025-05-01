package org.brain4j.math.precision;

import org.brain4j.math.constants.Constants;

import java.util.Arrays;

public final class PrecisionUtils {

    private PrecisionUtils() {}

    /**
     * Check if a double value can be accurately represented as a float without significant loss of precision.
     *
     * @param value the double value to check
     * @return true if the value can be represented as float, false otherwise
     */
    public static boolean canBeRepresentedAsFloat(double value) {
        if (Double.isNaN(value)) return Float.isNaN((float)value);
        if (Double.isInfinite(value)) return Float.isInfinite((float)value);

        if (value == 0.0d) return true;
        if (Math.abs(value) > Float.MAX_VALUE) return false;

        float asFloat = (float) value;
        if (Float.isInfinite(asFloat)) return false;

        double backToDouble = asFloat;
        if (value == backToDouble) return true;

        double absDiff = Math.abs(value - backToDouble);
        double absValue = Math.abs(value);

        if (absValue >= Float.MIN_NORMAL) {
            return absDiff / absValue < Constants.PRECISION_THRESHOLD;
        } else { // subnormal numbers case
            double absoluteErrorThreshold = Float.MIN_NORMAL * Constants.PRECISION_THRESHOLD;
            return absDiff < absoluteErrorThreshold;
        }
    }


    /**
     * Check if an array of double values requires double precision.
     *
     * @param data the double array to check
     * @return true if any value requires double precision, false otherwise
     */
    public static boolean requiresDoublePrecision(double[] data) {
        return Arrays.stream(data)
                .anyMatch(
                        value -> !canBeRepresentedAsFloat(value)
                );
    }

    /**
     * Get a string representation of the current precision mode.
     *
     * @param isUsingGPU whether GPU is being used
     * @param supportsFP64 whether FP64 is supported
     * @param requiresFP64 whether FP64 is required
     * @return a string describing the current precision mode
     */
    public static String getPrecisionModeDescription(
            boolean isUsingGPU,
            boolean supportsFP64,
            boolean requiresFP64
    ) {
        if (!isUsingGPU) {
            return "CPU (Full precision)";
        }

        if (supportsFP64) {
            return "GPU (Double precision supported)";
        }

        if (requiresFP64) {
            return "GPU with CPU fallback for precision-critical operations";
        }

        return "GPU (Single precision only)";
    }
}

