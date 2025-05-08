package org.brain4j.math;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.lang.reflect.Constructor;
import java.time.Duration;
import java.util.List;

import static org.brain4j.math.constants.Constants.GRADIENT_CLIP;

/**
 * Utility class for conversions and value matching.
 */
public class BrainUtils {

    public static float f16ToFloat(short half) {
        int sign = (half >> 15) & 0x1f;
        int exponent = (half >> 10) & 0x1f;
        int mantissa = half & 0x3ff;

        if (exponent == 31) {
            if (mantissa == 0) {
                // Inf
                return Float.intBitsToFloat((sign << 31) | 0x7f800000);
            } else {
                // NaN
                return Float.intBitsToFloat((sign << 31) | 0x7f800000 | (mantissa << 13));
            }
        }

        if (exponent == 0) {
            if (mantissa == 0) {
                // Zero
                return Float.intBitsToFloat(sign << 31);
            } else {
                // Subnormal -> normalize
                while ((mantissa & 0x00000400) == 0) {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                exponent += 1;
                mantissa &= ~0x00000400;
            }
        }

        exponent += (127 - 15);
        mantissa <<= 13;

        int result = (sign << 31) | (exponent << 23) | mantissa;
        return Float.intBitsToFloat(result);
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

    public static double estimateMaxLearningRate(Tensor X) {
        if (X.dimension() == 1) {
            throw new UnsupportedOperationException("Operation is not supported for 1D inputs yet!");
        }

        Tensor XtX = X.transpose().matmul(X);
        Tensor v = Tensors.random(XtX.shape()[1], 1);

        v = XtX.matmul(v);
        v = v.div(v.norm());

        double lambdaMax = v.transpose().matmul(XtX).matmul(v).get(0, 0);
        return 2.0 / lambdaMax;
    }

    public static <T extends Enum<T>> T parse(Tensor outputs, Class<T> clazz) {
        if (outputs.dimension() != 1) {
            throw new IllegalArgumentException("Output tensor must be 1-dimensional!");
        }

        return clazz.getEnumConstants()[argmax(outputs)];
    }

    public static String getHeader(String middleText, String character) {
        int maxLength = 70;
        int middleLength = middleText.length();

        String repeated = character.repeat((maxLength - middleLength) / 2);
        String result = repeated + middleText + repeated;

        return (result.length() != maxLength ? result + character : result) + "\n";
    }

    public static int argmax(Tensor input) {
        if (input.dimension() > 1) {
            throw new IllegalArgumentException("Input tensor must be 1-dimensional!");
        }
        
        int index = 0;
        double max = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < input.elements(); i++) {
            if (input.get(i) > max) {
                max = input.get(i);
                index = i;
            }
        }

        return index;
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

    public static double clipGradient(double gradient) {
        return Math.max(Math.min(gradient, GRADIENT_CLIP), -GRADIENT_CLIP);
    }

    public static String formatNumber(long params) {
        String[] prefixes = {"B", "KB", "MB", "GB", "TB"};

        if (params == 0) {
            return "0";
        }

        int ciphers = (int) (Math.log10(params) / 3);

        double divisor = Math.pow(1000, ciphers);
        double normalized = params / divisor;

        return "%.2f %s".formatted(normalized, prefixes[ciphers]);
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
}