package org.brain4j.common.constants;

/*
 * All constants are calculated through default strictfp directive to avoid precision loss
 */
public final class Constants {
    
    private Constants() {
    }

    public static final String RESET = "\033[0m";

    public static final String WHITE = "\033[97m";

    public static final String LIGHT_GREEN = "\033[32m";

    public static final String GREEN = "\033[92m";

    public static final String LIGHT_BLUE = "\033[34m";

    public static final String BLUE = "\033[94m";

    public static final String CYAN = "\033[96m";

    public static final String LIGHT_YELLOW = "\033[33m";

    public static final String YELLOW = "\033[93m";

    public static final String MAGENTA = "\033[95m";

    public static final String PURPLE = "\033[35m";

    public static final String GRAY = "\033[37m";

    public static final double PI = Math.PI;
    
    public static final double TWO_PI = 2.0 * PI;
    
    public static final double HALF_PI = PI / 2.0;
    
    public static final double QUARTER_PI = PI / 4.0;
    
    public static final double INV_PI = 1.0 / PI;

    public static final double E = Math.E;
    
    public static final double LN2 = Math.log(2.0);
    
    public static final double LN10 = Math.log(10.0);
    
    public static final double LOG2E = 1.0 / LN2;
    
    public static final double LOG10E = 1.0 / LN10;
    
    public static final double SQRT2 = Math.sqrt(2.0);

    public static final double SQRT3 = Math.sqrt(3.0);

    public static final double SQRT1_2 = 1.0 / Math.sqrt(2.0);
    
    public static final double GOLDEN_RATIO = (1.0 + Math.sqrt(5.0)) / 2.0;
    
    public static final double EULER_MASCHERONI = 0.57721566490153286061;
    
    public static final double EPSILON = 1.0e-10;

    public static final double RAD_TO_DEG = 180.0 / PI;
    
    public static final double DEG_TO_RAD = PI / 180.0;
    
    public static final double GRADIENT_CLIP = 5.0;
    
    public static final int FFT_THRESHOLD = 32;
    
    public static final int OPTIMAL_WORKGROUP_SIZE = 256;
    
    public static final int MAX_WORKGROUP_SIZE = 1024;

    public static final int WORK_THRESHOLD = 1024;

    public static final double PRECISION_THRESHOLD = 1e-7;
} 