__kernel void adam_update(
    __global float* firstMomentum, // First momentum vector
    __global float* secondMomentum, // Second momentum vector
    __global float* gradients, // Gradient vector
    __global float* updates, // Update vector
    float beta1, float beta2,
    float oneMinusBeta1, float oneMinusBeta2,
    float beta1Timestep, float beta2Timestep,
    float epsilon, float learningRate,
    int count) {

    int i = get_global_id(0);

    if (i < count) {
        float gradient = gradients[i];

        float m = beta1 * firstMomentum[i] + oneMinusBeta1 * gradient;
        float v = beta2 * secondMomentum[i] + oneMinusBeta2 * gradient * gradient;

        float mHat = m / beta1Timestep;
        float vHat = v / beta2Timestep;

        firstMomentum[i] = m;
        secondMomentum[i] = v;

        updates[i] = (learningRate * mHat) / (native_sqrt(vHat) + epsilon);
    }
}
