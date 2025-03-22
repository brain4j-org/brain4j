package net.echo.math4j.math.tensor;

import net.echo.math4j.math.tensor.autograd.AutogradContext;
import net.echo.math4j.math.vector.Vector;
import net.echo.math4j.math.tensor.index.Range;

import java.util.function.Function;
import java.util.function.Supplier;

public interface Tensor extends Iterable<Double> {

    AutogradContext getAutogradContext();

    void setAutogradContext(AutogradContext autogradContext);

    Vector getData();

    int[] shape();

    int dimension();

    int elements();

    Tensor set(double value, int... indices);

    float get(int... indices);

    Tensor add(double value, int... indices);

    Tensor add(Tensor other);

    Tensor add(double value);

    Tensor plus(Tensor other);

    Tensor plus(double value);

    Tensor sub(Tensor other);

    Tensor sub(double value);

    Tensor minus(Tensor other);

    Tensor minus(double value);

    Tensor mul(Tensor other);

    Tensor mul(double value);

    Tensor times(Tensor other);

    Tensor times(double value);

    Tensor div(Tensor other);

    Tensor div(double value);

    Tensor divide(Tensor other);

    Tensor divide(double value);

    /**
     * Performs a convolution between this tensor and the specified kernel tensor.
     * Implicitly uses SAME padding and FFT implementation for larger dimensions.
     * Convolution is supported for both 1D and 2D tensors.
     *
     * @param kernel the kernel tensor to use for convolution
     * @return a new tensor resulting from the convolution
     * @throws IllegalArgumentException if tensor dimensions are not compatible
     */
    Tensor convolve(Tensor kernel);

    double sum();

    double mean();

    double max();

    double min();

    double dot(Tensor other);

    double norm();

    double normSquared();

    Tensor normalize();

    double distance(Tensor other);

    double distanceSquared(Tensor other);

    Tensor map(Function<Double, Double> function);

    Tensor fill(double value);

    Tensor fill(Supplier<Double> supplier);

    float[] toArray();

    double[] toDoubleArray();

    Tensor reshape(int... newShape);

    Tensor transpose();

    Tensor permute(int... dims);

    Tensor select(int dim, int index);

    Tensor matmul(Tensor other);

    Tensor clone();

    Tensor mul(Vector vec);

    Tensor matmul(Vector vec);

    Tensor sum(int dim, boolean keepDim);

    Tensor mean(int dim, boolean keepDim);

    Tensor view(int... newShape);

    Tensor squeeze();

    Tensor squeeze(int dim);

    Tensor unsqueeze(int dim);

    Tensor slice(Range... ranges);

    Tensor requiresGrad(boolean requiresGrad);

    boolean requiresGrad();

    Tensor grad();

    void backward();

    void backward(Tensor gradOutput);

    Tensor addWithGrad(Tensor other);

    Tensor mulWithGrad(Tensor other);

    Tensor divWithGrad(Tensor other);

    Tensor subWithGrad(Tensor other);

    String toString(String format);

    Tensor softmax();

    Tensor gpu();
}