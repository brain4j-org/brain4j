package net.echo.math4j;

import static net.echo.math4j.math.constants.Constants.*;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.vector.Vector;

import java.lang.reflect.Constructor;
import java.util.List;

/**
 * Utility class for conversions and value matching.
 */
public class BrainUtils {

    public static <T extends Enum<T>> T parse(Vector outputs, Class<T> clazz) {
        return clazz.getEnumConstants()[argmax(outputs)];
    }

    public static String getHeader(String middleText) {
        String base = "=";

        int maxLength = 70;
        int middleLength = middleText.length();

        String repeated = base.repeat((maxLength - middleLength) / 2);
        String result = repeated + middleText + repeated;

        return (result.length() != maxLength ? result + "=" : result) + "\n";
    }
    
    public static int argmax(Vector inputs) {
        int index = 0;
        double max = inputs.get(0);

        for (int i = 1; i < inputs.size(); i++) {
            if (inputs.get(i) > max) {
                max = inputs.get(i);
                index = i;
            }
        }

        return index;
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

        return String.format("%.2f %s", normalized, prefixes[ciphers]);
    }

    @SuppressWarnings("all")
    public static <T> T newInstance(String classPath) throws Exception {
        Class<?> clazz = Class.forName(classPath);

        Constructor<?> constructor = clazz.getDeclaredConstructor();
        constructor.setAccessible(true);

        return (T) constructor.newInstance();
    }
}