package org.brain4j.common;

import org.brain4j.common.tensor.Tensor;

import java.lang.reflect.Constructor;
import java.time.Duration;
import java.util.List;

/**
 * General utilities class
 */
public class Commons {

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
                // subnormal / zero
                EXP_TABLE[i] = 0;
                OFF_TABLE[i] = 0;
            } else if (i == 31) {
                // inf/nan
                EXP_TABLE[i] = 0xFF << 23;
                OFF_TABLE[i] = 0;
            } else {
                // normale
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

    public static String createProgressBar(
        double percent,
        int characterCount,
        String barCharacter,
        String emptyCharacter
    ) {
        if (percent < 0 || percent > 1) {
            throw new IllegalArgumentException("Percent must be between 0 and 1!");
        }

        int fill = (int) Math.round(percent * characterCount);
        int remaining = characterCount - fill;

        return barCharacter.repeat(fill) + emptyCharacter.repeat(remaining);
    }

    public static float f16ToFloat(short half) {
        int bits = half & 0xFFFF;
        int sign = bits >>> 15;
        int exp = (bits >>> 10) & 0x1F;
        int mantissa = bits & 0x3FF;
        int floatBits = (sign << 31) | EXP_TABLE[exp] | MANT_TABLE[mantissa + OFF_TABLE[exp]];
        return Float.intBitsToFloat(floatBits);
    }

    public static int nextPowerOf2(int n) {
        if (n <= 0) return 1;
        n--;
        n |= n >>> 1;
        n |= n >>> 2;
        n |= n >>> 4;
        n |= n >>> 8;
        n |= n >>> 16;
        return n + 1;
    }

    public static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    public static String formatDuration(double seconds) {
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

        return (secs == 0)
                ? String.format("%dm", minutes)
                : String.format("%dm%ds", minutes, secs);
    }

    public static <T extends Enum<T>> T parse(Tensor outputs, Class<T> clazz) {
        if (outputs.rank() != 1) {
            throw new IllegalArgumentException("Output tensor must be 1-dimensional!");
        }

        return clazz.getEnumConstants()[outputs.argmax()];
    }

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

    public static String formatNumber(long params) {
        String[] prefixes = {"B", "KB", "MB", "GB", "TB"};

        if (params == 0) return "0";

        int exponent = (int) (Math.log10(params) / 3);

        double divisor = Math.pow(1000, exponent);
        double normalized = params / divisor;

        if (exponent > prefixes.length) {
            throw new UnsupportedOperationException("Input number too big!");
        }

        return "%.2f %s".formatted(normalized, prefixes[exponent]);
    }

    @SuppressWarnings("all")
    public static <T> T newInstance(String classPath) throws Exception {
        // Support for versions prior to 2.9
        classPath = classPath.replace("net.echo.brain4j", "org.brain4j.core");
        classPath = classPath.replace("net.echo.math", "org.brain4j.math");

        Class<?> clazz = Class.forName(classPath);

        Constructor<?> constructor = clazz.getDeclaredConstructor();
        constructor.setAccessible(true);

        return (T) constructor.newInstance();
    }

    public static double clamp(float value, double minimum, double maximum) {
        return Math.min(Math.max(value, minimum), maximum);
    }

    public static void waitAll(List<Thread> threads) {
        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace(System.err);
            }
        }
    }

    public static String getHeaderChar() {
        return "━";
    }
}