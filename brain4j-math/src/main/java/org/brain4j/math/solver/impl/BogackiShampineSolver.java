package org.brain4j.math.solver.impl;

import java.util.Arrays;
import java.util.function.Function;

import org.brain4j.math.solver.NumericalSolver;
import org.brain4j.math.solver.utils.StepResult;
import org.brain4j.math.tensor.Tensor;

/**
 * Third-order Bogacki-Shampine solver. Use it for cheap adaptive ODE steps with FSAL reuse.
 */
public class BogackiShampineSolver implements NumericalSolver {

    private static final float A21 = 1.0f / 2.0f;
    private static final float A32 = 3.0f / 4.0f;
    private static final float B1 = 2.0f / 9.0f;
    private static final float B2 = 1.0f / 3.0f;
    private static final float B3 = 4.0f / 9.0f;

    private static final float B1_HAT = 7.0f / 24.0f;
    private static final float B2_HAT = 1.0f / 4.0f;
    private static final float B3_HAT = 1.0f / 3.0f;
    private static final float B4_HAT = 1.0f / 8.0f;

    private static final float E1 = -5.0f / 72.0f;
    private static final float E2 = 1.0f / 12.0f;
    private static final float E3 = 1.0f / 9.0f;
    private static final float E4 = -1.0f / 8.0f;

    private static final float SAFETY_FACTOR = 0.9f;
    private static final float MIN_SCALE = 0.2f;
    private static final float MAX_SCALE = 5.0f;
    private static final float ORDER = 3.0f;

    private static final float STEP_SCALE_EXPONENT = 0.25f;
    private static final float MIN_ERROR_THRESHOLD = 1e-10f;

    private final ThreadLocal<Tensor> cachedK1 = new ThreadLocal<>();

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor update(
        Tensor deltaTimestep, Tensor tauT, Tensor projInput,
        Tensor hidden, Function<Tensor, Tensor> hiddenFunction
    ) {
        Tensor alpha = deltaTimestep.broadcastLike(tauT).div(tauT);

        Tensor k1;
        Tensor cached = cachedK1.get();

        if (cached != null && Arrays.equals(cached.shape(), hidden.shape())) {
            k1 = cached;
        } else {
            k1 = computeF(hidden, projInput, hiddenFunction);
        }

        Tensor k2 = computeF(hidden.add(alpha.times(k1).times(A21)), projInput, hiddenFunction);

        Tensor k3 = computeF(hidden.add(alpha.times(k2).times(A32)), projInput, hiddenFunction);

        Tensor increment = k1.times(B1).add(k2.times(B2)).add(k3.times(B3));

        Tensor nextHidden = hidden.add(alpha.times(increment));

        // FSAL reuses k4 as the next step k1.
        cachedK1.set(computeF(nextHidden, projInput, hiddenFunction));

        return nextHidden;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public StepResult updateAdaptive(
        Tensor deltaTimestep, Tensor tauT, Tensor projInput,
        Tensor hidden, Function<Tensor, Tensor> hiddenFunction,
        float tolerance
    ) {
        Tensor alpha = deltaTimestep.broadcastLike(tauT).div(tauT);

        Tensor k1;
        Tensor cached = cachedK1.get();

        if (cached != null && Arrays.equals(cached.shape(), hidden.shape())) {
            k1 = cached;
        } else {
            k1 = computeF(hidden, projInput, hiddenFunction);
        }

        Tensor k2 = computeF(hidden.add(alpha.times(k1).times(A21)), projInput, hiddenFunction);

        Tensor k3 = computeF(hidden.add(alpha.times(k2).times(A32)), projInput, hiddenFunction);

        Tensor increment = k1.times(B1).add(k2.times(B2)).add(k3.times(B3));

        Tensor nextHidden = hidden.add(alpha.times(increment));

        Tensor k4 = computeF(nextHidden, projInput, hiddenFunction);

        Tensor errorEstimate = k1.times(E1).add(k2.times(E2)).add(k3.times(E3)).add(k4.times(E4));

        Tensor scaledError = alpha.times(errorEstimate);

        // TODO: create a Tensor#maxAbs function
        float errorNorm = computeErrorNorm(scaledError);

        float scale;

        if (errorNorm < MIN_ERROR_THRESHOLD) {
            scale = MAX_SCALE;
        } else {
            scale = SAFETY_FACTOR * (float) Math.pow(tolerance / errorNorm, STEP_SCALE_EXPONENT);
        }

        scale = Math.clamp(scale, MIN_SCALE, MAX_SCALE);

        Tensor nextDeltaT = deltaTimestep.times(scale);

        if (errorNorm <= tolerance) {
            cachedK1.set(k4);

            return StepResult.accepted(nextHidden, nextDeltaT);
        } else {
            return StepResult.rejected(hidden, nextDeltaT);
        }
    }

    private float computeErrorNorm(Tensor error) {
        float maxError = 0.0f;
        int size = error.elements();

        for (int i = 0; i < size; i++) {
            float absValue = Math.abs(error.get(i));

            if (absValue > maxError) {
                maxError = absValue;
            }
        }

        return maxError;
    }

    private Tensor computeF(Tensor h, Tensor projInput, Function<Tensor, Tensor> hiddenFunction) {
        Tensor projHidden = hiddenFunction.apply(h);
        Tensor z = projInput.add(projHidden).tanh();

        return z.sub(h);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void resetCache() {
        cachedK1.remove();
    }
}
