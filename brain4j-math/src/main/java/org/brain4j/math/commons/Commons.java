package org.brain4j.math.commons;

import java.lang.reflect.Constructor;
import java.time.Duration;

import org.brain4j.math.tensor.Tensor;

import static java.util.Locale.ENGLISH;

/**
 * Shared math utilities. Use it for formatting, numerics, and tensor helpers.
 */
@SuppressWarnings("unchecked")
public class Commons {

    public static final String HEADER_CHAR = "━";
    private static final int[] EXP_TABLE = new int[64];
    private static final int[] MANT_TABLE = new int[2048];
    private static final int[] OFF_TABLE = new int[64];

    static {
        precomputeTables();
    }

    private static void precomputeTables() {
        for (int i = 0; i < 64; i++) {
            int e = i - 15;

            if (i == 0) {
                EXP_TABLE[i] = 0;
                OFF_TABLE[i] = 0;
            } else if (i == 31) {
                EXP_TABLE[i] = 0xFF << 23;
                OFF_TABLE[i] = 0;
            } else {
                EXP_TABLE[i] = (e + 127) << 23;
                OFF_TABLE[i] = 1024;
            }
        }

        MANT_TABLE[0] = 0;

        for (int i = 1; i < 2048; i++) {
            int m = i & 0x3FF;
            int e = i >> 10;

            if (e == 0) {
                int mantissa = m;
                int exp = -1;

                do {
                    mantissa <<= 1;
                    exp--;
                } while ((mantissa & 0x400) == 0);

                mantissa &= 0x3FF;
                MANT_TABLE[i] = (mantissa << 13) | ((exp + 1 + 127) << 23);
            } else {
                MANT_TABLE[i] = m << 13;
            }
        }
    }

    /**
     * Build a textual progress bar.
     *
     * @param percent progress in [0, 1]
     * @return the rendered bar
     * @throws IllegalArgumentException if percent is outside [0, 1]
     */
    public static String createProgressBar(
        double percent,
        int characterCount,
        String barPrefix,
        String barCharacter,
        String emptyPrefix,
        String emptyCharacter
    ) {
        if (percent < 0 || percent > 1) {
            throw new IllegalArgumentException("Percent must be between 0 and 1!");
        }

        int fill = (int) Math.round(percent * characterCount);
        int remaining = characterCount - fill;

        return barPrefix + barCharacter.repeat(fill) + emptyPrefix + emptyCharacter.repeat(remaining);
    }

    /**
     * Convert an FP16 value to float.
     *
     * @return the 32-bit value
     */
    public static float f16ToFloat(short half) {
        int bits = half & 0xFFFF;
        int sign = bits >>> 15;
        int exp = (bits >>> 10) & 0b11111;
        int mantissa = bits & 0b1111111111;
        int floatBits = (sign << 31) | EXP_TABLE[exp] | MANT_TABLE[mantissa + OFF_TABLE[exp]];

        return Float.intBitsToFloat(floatBits);
    }

    /**
     * Return the next power of two.
     *
     * @return the smallest power of two at or above n
     */
    public static int nextPowerOf2(int n) {
        if (n <= 0) {
            return 1;
        }

        n--;
        n |= n >>> 1;
        n |= n >>> 2;
        n |= n >>> 4;
        n |= n >>> 8;
        n |= n >>> 16;

        return n + 1;
    }

    /**
     * Check for a power of two.
     *
     * @return true for positive powers of two
     */
    public static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    /**
     * Format seconds as a readable duration.
     *
     * @return the formatted duration
     */
    public static String formatDuration(double seconds) {
        if (seconds < 0) {
            return "N/A";
        }

        double millis = seconds * 1000;
        Duration duration = Duration.ofMillis((long) millis);

        if (seconds < 1) {
            return String.format("%.2fms", millis);
        }

        if (seconds < 60) {
            return String.format("%.2fs", seconds);
        }

        long minutes = duration.toMinutesPart();
        long secs = duration.toSecondsPart();
        long hours = duration.toHoursPart();

        if (hours < 1) {
            return (secs == 0) ? String.format("%dm", minutes) : String.format("%dm%ds", minutes, secs);
        }

        return (minutes == 0) ? String.format("%dh", hours) : String.format("%dh%dm", hours, minutes);
    }

    /**
     * Parse a 1D output tensor into an enum constant.
     *
     * @param outputs 1D model outputs
     * @return the argmax enum constant
     * @throws IllegalArgumentException if outputs is not 1-dimensional
     */
    public static <T extends Enum<T>> T parse(Tensor outputs, Class<T> clazz) {
        if (outputs.rank() != 1) {
            throw new IllegalArgumentException("Output tensor must be 1-dimensional!");
        }

        return clazz.getEnumConstants()[outputs.argmax()];
    }

    /**
     * Build a centered header line.
     *
     * @return the formatted header
     */
    public static String getHeader(String middleText, String character) {
        int maxLength = 70;
        int middleLength = middleText.length();
        int repeatedLength = (maxLength - middleLength) / 2;

        String repeated = character.repeat(repeatedLength);
        String result = repeated + middleText + repeated;

        if (result.length() < maxLength) {
            result += character;
        }

        return result + "\n";
    }

    /**
     * Format a value with SI decimal prefixes.
     *
     * @param bytes non-negative value in bytes
     * @return the scaled string
     * @throws UnsupportedOperationException if the value exceeds TB
     */
    public static String formatNumber(long bytes) {
        String[] prefixes = {"B", "KB", "MB", "GB", "TB"};

        if (bytes == 0) {
            return "0";
        }

        int exponent = (int) (Math.log10(bytes) / 3);

        double divisor = Math.pow(1000, exponent);
        double normalized = bytes / divisor;

        if (exponent >= prefixes.length) {
            throw Commons.illegalArgument("Input number is too big to be parsed!");
        }

        return "%.2f %s".formatted(normalized, prefixes[exponent]);
    }

    /**
     * Create an instance from a class name.
     *
     * @return a new instance
     */
    @SuppressWarnings("all")
    public static <T> T newInstance(String classPath) {
        // Keep backward compatibility with pre-2.9 package names.
        classPath = classPath.replace("net.echo.brain4j", "org.brain4j.core");
        classPath = classPath.replace("net.echo.math", "org.brain4j.math");

        try {
            Class<?> clazz = Class.forName(classPath);

            return (T) newInstance(clazz);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create an instance with the no-arg constructor.
     *
     * @return a new instance
     */
    @SuppressWarnings("all")
    public static <T> T newInstance(Class<T> clazz) {
        try {
            Constructor<?> constructor = clazz.getDeclaredConstructor();
            constructor.setAccessible(true);

            return (T) constructor.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Convert ints to floats.
     *
     * @return a new float array
     */
    public static float[] int2float(int[] array) {
        float[] result = new float[array.length];

        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }

        return result;
    }

    /**
     * Compute a non-negative modulo.
     *
     * @param m positive modulus
     * @return the result in [0, m)
     */
    public static int mod(int x, int m) {
        int r = x % m;

        return r < 0 ? r + m : r;
    }

    /**
     * Build an illegal-argument exception.
     *
     * @return the exception to throw
     */
    public static IllegalArgumentException illegalArgument(String message, Object... args) {
        return new IllegalArgumentException(String.format(message, args));
    }

    /**
     * Build an illegal-state exception.
     *
     * @return the exception to throw
     */
    public static IllegalStateException illegalState(String message, Object... args) {
        return new IllegalStateException(String.format(message, args));
    }

    /**
     * Build an index-out-of-bounds exception.
     *
     * @return the exception to throw
     */
    public static IndexOutOfBoundsException indexOOB(String message, Object... args) {
        return new IndexOutOfBoundsException(String.format(message, args));
    }

    /**
     * Capitalize the first character.
     *
     * @return the capitalized text
     */
    public static String capitalize(String text) {
        if (text == null || text.isEmpty()) {
            return text;
        }

        return text.substring(0, 1).toUpperCase(ENGLISH) + text.substring(1);
    }
}
