package org.brain4j.math.commons;

import org.brain4j.math.tensor.Tensor;

import java.lang.reflect.Constructor;
import java.time.Duration;

import static java.util.Locale.ENGLISH;

/**
 * General utility methods used across the math module.
 *
 * <p>This class provides various helper functions including:
 * <ul>
 *   <li>Progress bar and formatting utilities for training output
 *   <li>Numeric utilities (clamp, modulo, float16 conversion)
 *   <li>Reflection helpers for instantiating classes
 *   <li>Tensor classification helpers
 * </ul>
 *
 * <p>Many methods in this class are used internally by the framework's
 * training and evaluation components.
 *
 * @author xEcho1337
 */
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
     * Creates a textual progress bar representation.
     * <p>
     * The progress is expressed as a value in the range {@code [0, 1]} and mapped
     * to a fixed number of characters. The filled portion is rendered using
     * {@code barCharacter}, while the remaining portion uses {@code emptyCharacter}.
     *
     * @param percent progress value in the range {@code [0, 1]}
     * @param characterCount total number of characters composing the bar
     * @param barCharacter character used for the filled portion
     * @param emptyCharacter character used for the empty portion
     * @return a string representing the progress bar
     * @throws IllegalArgumentException if {@code percent} is outside {@code [0, 1]}
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
     * Converts a 16-bit IEEE 754 half-precision floating-point value (FP16)
     * into a 32-bit single-precision float.
     * <p>
     * This implementation uses precomputed lookup tables to efficiently handle
     * normal, subnormal, zero, infinity, and NaN values without branching.
     *
     * @param half the half-precision floating-point value encoded as a {@code short}
     * @return the corresponding 32-bit floating-point value
     */
    public static float f16ToFloat(short half) {
        int bits = half & 0xFFFF;
        // [1 bit sign] | [5 bits exponent] | [12 bits mantissa]
        int sign = bits >>> 15; // 1 bit of sign
        int exp = (bits >>> 10) & 0b11111; // 5 bits of exp
        int mantissa = bits & 0b1111111111; // 12 bits of mantissa
        int floatBits = (sign << 31) | EXP_TABLE[exp] | MANT_TABLE[mantissa + OFF_TABLE[exp]];
        return Float.intBitsToFloat(floatBits);
    }
    
    /**
     * Computes the smallest power of two greater than or equal to the given value.
     * <p>
     * If the input is less than or equal to zero, the method returns {@code 1}.
     *
     * @param n the input value
     * @return the next power of two greater than or equal to {@code n}
     */
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
    
    /**
     * Checks whether the given integer is a power of two.
     *
     * @param n the value to check
     * @return {@code true} if {@code n} is a positive power of two, {@code false} otherwise
     */
    public static boolean isPowerOf2(int n) {
        // bit shifting tricks: it's a power of 2 only if one bit is 1
        // ex: 4 & 3 = 0b100 & 0b011 = 0b000 = 0
        // ex: 5 & 4 = 0b101 & 0b100 = 0b100 = 4
        return n > 0 && (n & (n - 1)) == 0;
    }
    
    /**
     * Formats a duration expressed in seconds into a human-readable string.
     * <p>
     * The output format depends on the magnitude of the duration:
     * <ul>
     *   <li>Less than 1 second: milliseconds (e.g. {@code 12.34ms})</li>
     *   <li>Less than 60 seconds: seconds (e.g. {@code 1.23s})</li>
     *   <li>Less than 1 hour: minutes and seconds (e.g. {@code 2m15s})</li>
     *   <li>One minute or more: hours and minutes (e.g. {@code 4h32m})</li>
     * </ul>
     *
     * @param seconds the duration in seconds
     * @return a formatted string representing the duration
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
     * Parses a classification output tensor into an enum constant.
     * <p>
     * Takes a 1D tensor of class probabilities/logits and returns the enum constant
     * corresponding to the highest score (argmax). Useful for classification tasks
     * where the output classes are represented by an enum.
     *
     * @param outputs the model output tensor (must be 1D)
     * @param clazz the enum class containing the possible classes
     * @param <T> the enum type
     * @return the predicted enum class
     * @throws IllegalArgumentException if outputs is not 1-dimensional
     */
    public static <T extends Enum<T>> T parse(Tensor outputs, Class<T> clazz) {
        if (outputs.rank() != 1) {
            throw new IllegalArgumentException("Output tensor must be 1-dimensional!");
        }

        return clazz.getEnumConstants()[outputs.argmax()];
    }

    /**
     * Creates a header string with centered text.
     * <p>
     * The text is centered between repeating characters, useful for
     * creating section headers in console output.
     *
     * @param middleText the text to center
     * @param character the character to repeat around the text
     * @return a formatted header string
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
     * Formats a positive numeric value using SI decimal prefixes (base 1000),
     * producing a human-readable string with two decimal digits.
     * <p>
     * The following prefixes are supported:
     * <ul>
     *   <li>B  (bytes)</li>
     *   <li>KB (10³)</li>
     *   <li>MB (10⁶)</li>
     *   <li>GB (10⁹)</li>
     *   <li>TB (10¹²)</li>
     * </ul>
     *
     * The method assumes a non-negative input value. A value of {@code 0}
     * is formatted as {@code "0"} without any suffix.
     *
     * @param bytes the numeric value to format, expressed in bytes
     * @return a formatted string representing the value scaled with an SI prefix
     * @throws UnsupportedOperationException if the value exceeds the largest
     *         supported prefix (TB)
     */
    public static String formatNumber(long bytes) {
        String[] prefixes = {"B", "KB", "MB", "GB", "TB"};

        if (bytes == 0) return "0";

        int exponent = (int) (Math.log10(bytes) / 3);

        double divisor = Math.pow(1000, exponent);
        double normalized = bytes / divisor; // ex. 12345 -> 1.2345

        if (exponent >= prefixes.length) {
            throw Commons.illegalArgument("Input number is too big to be parsed!");
        }

        return "%.2f %s".formatted(normalized, prefixes[exponent]);
    }
    
    /**
     * Creates a new instance of a class given its fully qualified name.
     * <p>
     * This method provides backward compatibility by remapping legacy package
     * names to their current equivalents before class loading.
     *
     * @param classPath the fully qualified class name
     * @param <T> the expected type of the created instance
     * @return a new instance of the specified class
     * @throws RuntimeException if the class cannot be found or instantiated
     */
    @SuppressWarnings("all")
    public static <T> T newInstance(String classPath) {
        // Support for versions prior to 2.9
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
     * Creates a new instance of the specified class using its no-argument constructor.
     * <p>
     * The constructor is made accessible if necessary.
     *
     * @param clazz the class to instantiate
     * @param <T> the type of the class
     * @return a new instance of {@code clazz}
     * @throws RuntimeException if instantiation fails
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
     * Converts an integer array to a float array by widening each element.
     *
     * @param array the input integer array
     * @return a new array containing the converted floating-point values
     */
    public static float[] int2float(int[] array) {
        float[] result = new float[array.length];

        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }

        return result;
    }
    
    /**
     * Computes a mathematical modulo operation with a non-negative result.
     * <p>
     * Unlike the Java remainder operator ({@code %}), this method guarantees
     * a result in the range {@code [0, m)} for {@code m > 0}.
     *
     * @param x the dividend
     * @param m the modulus
     * @return the modulo result in the range {@code [0, m)}
     */
    public static int mod(int x, int m) {
        int r = x % m;
        return r < 0 ? r + m : r;
    }
    
    /**
     * Returns an {@link IllegalArgumentException} with a formatted message.
     *
     * @param message the exception message format string
     * @param args arguments referenced by the format specifiers
     * @return IllegalArgumentException the exception to throw
     */
    public static IllegalArgumentException illegalArgument(String message, Object... args) {
        return new IllegalArgumentException(String.format(message, args));
    }
    
    /**
     * Returns an {@link IllegalStateException} with a formatted message.
     *
     * @param message the exception message format string
     * @param args arguments referenced by the format specifiers
     * @throws IllegalStateException the exception to throw
     */
    public static IllegalStateException illegalState(String message, Object... args) {
        return new IllegalStateException(String.format(message, args));
    }
    
    /**
     * Returns an {@link IndexOutOfBoundsException} with a formatted message.
     * @param message the exception message format string
     * @param args arguments referenced by the format specifiers
     * @return the exception to throw
     */
    public static IndexOutOfBoundsException indexOOB(String message, Object... args) {
        return new IndexOutOfBoundsException(String.format(message, args));
    }
    
    public static String capitalize(String text) {
        if (text == null || text.isEmpty()) {
            return text;
        }
        return text.substring(0, 1).toUpperCase(ENGLISH) + text.substring(1);
    }
}