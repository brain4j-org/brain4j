__kernel void adam_update(
    __global double* firstMomentum, // First momentum vector
    __global double* secondMomentum, // Second momentum vector
    __global double* gradients, // Gradient vector
    __global double* updates, // Update vector
    double beta1, double beta2,
    double oneMinusBeta1, double oneMinusBeta2,
    double beta1Timestep, double beta2Timestep,
    double epsilon, double learningRate,
    int count) {

    int i = get_global_id(0);
    if (i < count) {
        double gradient = gradients[i];

        double m = beta1 * firstMomentum[i] + oneMinusBeta1 * gradient;
        double v = beta2 * secondMomentum[i] + oneMinusBeta2 * gradient * gradient;

        firstMomentum[i] = m;
        secondMomentum[i] = v;

        double mHat = m / beta1Timestep;
        double vHat = v / beta2Timestep;

        updates[i] = (learningRate * mHat) / (native_sqrt(vHat) + epsilon);
    }
}
