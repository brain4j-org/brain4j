package org.brain4j.math.solver.impl;

import org.brain4j.math.solver.NumericalSolver;
import org.brain4j.math.solver.utils.StepResult;
import org.brain4j.math.tensor.Tensor;

import java.util.Arrays;
import java.util.function.Function;

/**
 * Implements the Bogacki–Shampine method for integrating ordinary differential
 * equations (ODEs) of the form:
 *
 * <pre>
 *     dh/dt = F(h, t)
 * </pre>
 *
 * The Bogacki–Shampine method is a third-order Runge–Kutta method with four stages
 * and the First Same As Last (FSAL) property, requiring only approximately three
 * function evaluations per step. It was proposed by Przemysław Bogacki and
 * Lawrence F. Shampine in 1989.
 *
 * <p>The Butcher tableau for the Bogacki–Shampine method is:</p>
 * <pre>
 *   0   |
 *  1/2  | 1/2
 *  3/4  |  0   3/4
 *   1   | 2/9  1/3  4/9
 *  -----|------------------
 *       | 2/9  1/3  4/9   0   (3rd order)
 *       | 7/24 1/4  1/3  1/8  (2nd order, for error estimation)
 * </pre>
 *
 * <p>The method computes:</p>
 * <pre>
 *   k1 = F(h, t)
 *   k2 = F(h + 0.5*Δt*k1, t + Δt/2)
 *   k3 = F(h + 0.75*Δt*k2, t + 3Δt/4)
 *   h_next = h + Δt * (2/9*k1 + 1/3*k2 + 4/9*k3)
 *   k4 = F(h_next, t + Δt)  // FSAL: becomes k1 in next step
 * </pre>
 *
 * <p>Low-order methods like Bogacki–Shampine are more suitable than higher-order
 * methods (e.g., Dormand–Prince) when only a crude approximation is required.
 * This method outperforms other third-order methods with an embedded second-order method.</p>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * NumericalSolver solver = new BogackiShampineSolver();
 * Tensor nextHidden = solver.update(deltaT, tau, projInput, hidden, x -> hiddenParams.forward(cache, x));
 * }</pre>
 *
 * @author Adversing
 * @see EulerSolver
 * @see RungeKuttaSolver
 */
public class BogackiShampineSolver implements NumericalSolver {

    // Butcher tableau coefficients for third-order solution
    private static final float A21 = 1.0f / 2.0f;
    private static final float A32 = 3.0f / 4.0f;
    private static final float B1 = 2.0f / 9.0f;
    private static final float B2 = 1.0f / 3.0f;
    private static final float B3 = 4.0f / 9.0f;

    // embedded second-order coefficients for error estimation
    private static final float B1_HAT = 7.0f / 24.0f;
    private static final float B2_HAT = 1.0f / 4.0f;
    private static final float B3_HAT = 1.0f / 3.0f;
    private static final float B4_HAT = 1.0f / 8.0f;

    // error coefficients: E_i = b_i - b_i_hat
    private static final float E1 = -5.0f / 72.0f;     // 2/9 - 7/24 = -5/72
    private static final float E2 = 1.0f / 12.0f;      // 1/3 - 1/4 = 1/12
    private static final float E3 = 1.0f / 9.0f;       // 4/9 - 1/3 = 1/9
    private static final float E4 = -1.0f / 8.0f;      // 0 - 1/8 = -1/8

    // step size control parameters
    private static final float SAFETY_FACTOR = 0.9f;
    private static final float MIN_SCALE = 0.2f;
    private static final float MAX_SCALE = 5.0f;
    private static final float ORDER = 3.0f;

    // 1/(order+1) for a third-order method = 0.25f
    private static final float STEP_SCALE_EXPONENT = 0.25f;
    private static final float MIN_ERROR_THRESHOLD = 1e-10f;

    // cache for FSAL property - k4 from previous step becomes k1 in next step
    private final ThreadLocal<Tensor> cachedK1 = new ThreadLocal<>();

    @Override
    public Tensor update(
        Tensor deltaTimestep, Tensor tauT, Tensor projInput,
        Tensor hidden, Function<Tensor, Tensor> hiddenFunction
    ) {
        // Δt / τ(x)
        Tensor alpha = deltaTimestep.broadcastLike(tauT).divGrad(tauT);

        // k1 = F(h_n, t_n)
        // let's use the FSAL property: if we have cached k1 from previous step, use it
        Tensor k1;
        Tensor cached = cachedK1.get();
        if (cached != null && Arrays.equals(cached.shape(), hidden.shape())) {
            k1 = cached;
        } else {
            k1 = computeF(hidden, projInput, hiddenFunction);
        }

        // k2 = F(h_n + 0.5*Δt*k1, t_n + Δt/2)
        Tensor k2 = computeF(hidden.addGrad(alpha.times(k1).times(A21)), projInput, hiddenFunction);

        // k3 = F(h_n + 0.75*Δt*k2, t_n + 3Δt/4)
        Tensor k3 = computeF(hidden.addGrad(alpha.times(k2).times(A32)), projInput, hiddenFunction);

        // h_{n+1} = h_n + Δt * (2/9*k1 + 1/3*k2 + 4/9*k3)
        Tensor increment = k1.times(B1).addGrad(k2.times(B2)).addGrad(k3.times(B3));

        Tensor nextHidden = hidden.addGrad(alpha.times(increment));

        // k4 = F(h_{n+1}, t_{n+1}); cache for FSAL property
        cachedK1.set(computeF(nextHidden, projInput, hiddenFunction));

        return nextHidden;
    }

    @Override
    public StepResult updateAdaptive(
        Tensor deltaTimestep, Tensor tauT, Tensor projInput,
        Tensor hidden, Function<Tensor, Tensor> hiddenFunction,
        float tolerance
    ) {
        // Δt / τ(x)
        Tensor alpha = deltaTimestep.broadcastLike(tauT).divGrad(tauT);

        // k1 = F(h_n, t_n) - use FSAL property if available
        Tensor k1;
        Tensor cached = cachedK1.get();
        if (cached != null && Arrays.equals(cached.shape(), hidden.shape())) {
            k1 = cached;
        } else {
            k1 = computeF(hidden, projInput, hiddenFunction);
        }

        // k2 = F(h_n + 0.5*Δt*k1, t_n + Δt/2)
        Tensor k2 = computeF(hidden.addGrad(alpha.times(k1).times(A21)), projInput, hiddenFunction);

        // k3 = F(h_n + 0.75*Δt*k2, t_n + 3Δt/4)
        Tensor k3 = computeF(hidden.addGrad(alpha.times(k2).times(A32)), projInput, hiddenFunction);

        // h_{n+1} = h_n + Δt * (2/9*k1 + 1/3*k2 + 4/9*k3) - third-order solution
        Tensor increment = k1.times(B1).addGrad(k2.times(B2)).addGrad(k3.times(B3));

        Tensor nextHidden = hidden.addGrad(alpha.times(increment));

        // k4 = F(h_{n+1}, t_{n+1})
        Tensor k4 = computeF(nextHidden, projInput, hiddenFunction);

        // err = Δt * (E1*k1 + E2*k2 + E3*k3 + E4*k4)
        Tensor errorEstimate = k1.times(E1).addGrad(k2.times(E2)).addGrad(k3.times(E3)).addGrad(k4.times(E4));

        Tensor scaledError = alpha.times(errorEstimate);

        // error norm (max absolute value across all elements)
        // TODO: create a Tensor#maxAbs function
        float errorNorm = computeErrorNorm(scaledError);

        // dt_new = dt * safety * (tol / err)^(1/(order+1))
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

    /**
     * Computes the maximum absolute error norm across all elements of the tensor.
     *
     * @param error the error tensor
     * @return the maximum absolute value in the tensor
     */
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

    /**
     * Computes the derivative function F(h, t) = tanh(Wx + Uh + b) - h
     * where the activation represents the target state and h is the current state.
     *
     * @param h              the current hidden state
     * @param projInput      the projected input (Wx + b)
     * @param hiddenFunction function to project the hidden state (Uh)
     * @return the derivative F(h, t)
     */
    private Tensor computeF(Tensor h, Tensor projInput, Function<Tensor, Tensor> hiddenFunction) {
        Tensor projHidden = hiddenFunction.apply(h);
        Tensor z = projInput.addGrad(projHidden).tanh();
        return z.subGrad(h);
    }

    @Override
    public void resetCache() {
        cachedK1.remove();
    }
}

