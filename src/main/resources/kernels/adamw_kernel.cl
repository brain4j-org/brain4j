__kernel void adam_update(
    __global double* firstMomentum,
    __global double* secondMomentum,
    __global double* gradients,
    __global double* updates,
    __global double* weights,
    double weightDecay,
    double beta1, double beta2,
    double beta1Timestep, double beta2Timestep,
    double epsilon, double learningRate,
    int count) {

    int i = get_global_id(0);
    if (i < count) {
        double gradient = gradients[i];

        double m = beta1 * firstMomentum[i] + (1 - beta1) * gradient;
        double v = beta2 * secondMomentum[i] + (1 - beta2) * gradient * gradient;

        firstMomentum[i] = m;
        secondMomentum[i] = v;

        double mHat = m / (1 - beta1Timestep);
        double vHat = v / (1 - beta2Timestep);

        updates[i] = (learningRate * mHat) / (sqrt(vHat) + epsilon) + weightDecay * weights[i];
    }
}
