package org.brain4j.math.solver.impl;

import java.util.function.Function;

import org.brain4j.math.activation.Activations;
import org.brain4j.math.solver.NumericalSolver;
import org.brain4j.math.tensor.Tensor;

/**
 * Explicit Euler solver. Use it for fast crude ODE steps with optional substepping.
 */
public record EulerSolver(int mSteps) implements NumericalSolver {

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor update(Tensor deltaTimestep, Tensor tauT, Tensor projInput, Tensor hidden, Function<Tensor, Tensor> hiddenFunction) {
        Tensor deltaNorm = deltaTimestep.divide(mSteps).withGrad();
        Tensor deltaTime = deltaNorm.broadcastLike(tauT).div(tauT);

        for (int i = 0; i < mSteps; i++) {
            Tensor projHidden = hiddenFunction.apply(hidden);
            Tensor z = projInput.add(projHidden).activate(Activations.TANH.function());
            Tensor deltaH = z.add(hidden.mul(-1));

            hidden = hidden.add(deltaTime.mul(deltaH));
        }

        return hidden;
    }
}
