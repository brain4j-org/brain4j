package tensor;

import net.echo.brain4j.utils.math.tensor.Tensor;
import net.echo.brain4j.utils.math.tensor.index.Range;

public class TensorExample {

    public static void main(String[] args) {
        System.out.println("Tensor Class Example:");
        
        testXORNeuralNetwork();
        testTensorOperations();
        testAdvancedIndexing();
        testReductionOperations();
        testReshapeOperations();
        testAutogradFeatures();
    }
    
    private static void testXORNeuralNetwork() {
        System.out.println("\n1. XOR Neural Network:");
        
        int inputSize = 2;
        int hiddenSize = 3;
        int outputSize = 1;
        
        Tensor[] inputs = {
            Tensor.vector(0, 0),
            Tensor.vector(0, 1),
            Tensor.vector(1, 0),
            Tensor.vector(1, 1)
        };
        
        Tensor[] labels = {
            Tensor.vector(0),
            Tensor.vector(1),
            Tensor.vector(1),
            Tensor.vector(0)
        };
        
        Tensor W1 = Tensor.randn(0.0, 0.5, hiddenSize, inputSize);
        Tensor b1 = Tensor.zeros(hiddenSize);
        
        Tensor W2 = Tensor.randn(0.0, 0.5, outputSize, hiddenSize);
        Tensor b2 = Tensor.zeros(outputSize);
        
        double learningRate = 0.1;
        int epochs = 10000;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            
            for (int i = 0; i < inputs.length; i++) {
                Tensor hidden = W1.matmul(inputs[i].reshape(inputSize, 1));
                hidden.add(b1.reshape(hiddenSize, 1));
                
                Tensor hiddenActivated = hidden.clone().map(x -> Math.max(0, x));
                
                Tensor output = W2.matmul(hiddenActivated);
                output.add(b2.reshape(outputSize, 1));
                
                Tensor predicted = output.clone().map(x -> 1.0 / (1.0 + Math.exp(-x)));
                
                Tensor error = predicted.minus(labels[i].reshape(outputSize, 1));
                double loss = error.normSquared();
                totalLoss += loss;
                
                Tensor gradOutput = error.times(predicted.map(x -> x * (1 - x)));
                
                Tensor gradW2 = gradOutput.matmul(hiddenActivated.transpose());
                W2.sub(gradW2.times(learningRate));
                b2.sub(gradOutput.reshape(outputSize).times(learningRate));
                
                Tensor gradHidden = W2.transpose().matmul(gradOutput);
                
                Tensor gradHiddenActivated = gradHidden.clone();
                for (int j = 0; j < hiddenSize; j++) {
                    if (hidden.get(j, 0) <= 0) {
                        gradHiddenActivated.set(0, j, 0);
                    }
                }
                
                Tensor gradW1 = gradHiddenActivated.matmul(inputs[i].reshape(1, inputSize));
                W1.sub(gradW1.times(learningRate));
                b1.sub(gradHiddenActivated.reshape(hiddenSize).times(learningRate));
            }
            
            if ((epoch + 1) % 1000 == 0) {
                System.out.printf("Epoch %d: Average Loss = %.6f%n", epoch + 1, totalLoss / inputs.length);
            }
        }
        
        System.out.println("\nPredictions after training:");
        
        for (int i = 0; i < inputs.length; i++) {
            Tensor hidden = W1.matmul(inputs[i].reshape(inputSize, 1));
            hidden.add(b1.reshape(hiddenSize, 1));
            
            Tensor hiddenActivated = hidden.clone().map(x -> Math.max(0, x));
            
            Tensor output = W2.matmul(hiddenActivated);
            output.add(b2.reshape(outputSize, 1));
            
            Tensor predicted = output.map(x -> 1.0 / (1.0 + Math.exp(-x)));
            
            System.out.printf("Input: [%.0f, %.0f], Output: %.6f, Expected: %.0f%n",
                    inputs[i].get(0), inputs[i].get(1), 
                    predicted.get(0, 0), labels[i].get(0));
        }
    }
    
    private static void testTensorOperations() {
        System.out.println("\n2. Basic Tensor Operations:");
        
        Tensor matrix = Tensor.matrix(2, 3, 1, 2, 3, 4, 5, 6);
        System.out.println("2x3 Matrix:\n" + matrix);
        
        Tensor transposed = matrix.transpose();
        System.out.println("3x2 Transposed:\n" + transposed);
        
        Tensor reshaped = matrix.reshape(1, 6);
        System.out.println("Reshaped to 1x6:\n" + reshaped);
        
        Tensor a = Tensor.ones(2, 2);
        Tensor b = Tensor.vector(2, 3).reshape(2, 1);
        System.out.println("a (ones matrix):\n" + a);
        System.out.println("b (column vector):\n" + b);
        
        Tensor c = a.times(b);
        System.out.println("a * b (element-wise with broadcasting):\n" + c);
        
        Tensor d = a.matmul(b);
        System.out.println("a · b (matrix-vector product):\n" + d);
        
        Tensor e = Tensor.matrix(2, 2, 1, 2, 3, 4);
        Tensor f = Tensor.matrix(2, 2, 5, 6, 7, 8);
        System.out.println("e:\n" + e);
        System.out.println("f:\n" + f);
        
        System.out.println("e + f:\n" + e.plus(f));
        System.out.println("e - f:\n" + e.minus(f));
        System.out.println("e * f (element-wise):\n" + e.times(f));
        System.out.println("e · f (matrix product):\n" + e.matmul(f));
        
        System.out.println("e * 2.0:\n" + e.times(2.0));
        System.out.println("f / 2.0:\n" + f.divide(2.0));
    }
    
    private static void testAdvancedIndexing() {
        System.out.println("\n3. Advanced Indexing:");
        
        Tensor t = Tensor.of(new int[]{3, 4}, 
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        );
        System.out.println("Original 3x4 tensor:\n" + t);
        
        Range r1 = new Range(0, 2);
        Range r2 = new Range(1, 3);
        
        Tensor sliced = t.slice(r1, r2);
        System.out.println("Sliced t[0:2, 1:3]:\n" + sliced);
        
        Tensor singleRow = t.select(0, 1);
        System.out.println("Second row of t:\n" + singleRow);
        
        Tensor singleCol = t.select(1, 2);
        System.out.println("Third column of t:\n" + singleCol);
        
        Range r3 = new Range(0, 3, 2);
        Tensor strided = t.slice(r3, null);
        System.out.println("Strided t[0:3:2, :]:\n" + strided);
    }
    
    private static void testReductionOperations() {
        System.out.println("\n4. Reduction Operations:");
        
        Tensor t = Tensor.of(new int[]{2, 3},
            1, 2, 3,
            4, 5, 6
        );
        System.out.println("Tensor t:\n" + t);
        
        System.out.println("Sum of all elements: " + t.sum());
        System.out.println("Mean of all elements: " + t.mean());
        System.out.println("Max value: " + t.max());
        System.out.println("Min value: " + t.min());
        
        Tensor rowSum = t.sum(1, false);
        System.out.println("Sum along rows: " + rowSum);
        
        Tensor colSum = t.sum(0, false);
        System.out.println("Sum along columns: " + colSum);
        
        Tensor rowMean = t.mean(1, false);
        System.out.println("Mean along rows: " + rowMean);
        
        Tensor colMean = t.mean(0, true);
        System.out.println("Mean along columns (keepDim=true):\n" + colMean);
    }
    
    private static void testReshapeOperations() {
        System.out.println("\n5. Reshape Operations:");
        
        Tensor t = Tensor.of(new int[]{2, 3},
            1, 2, 3,
            4, 5, 6
        );
        System.out.println("Original tensor t:\n" + t);
        
        Tensor reshaped = t.reshape(3, 2);
        System.out.println("Reshaped to 3x2:\n" + reshaped);
        
        Tensor viewed = t.view(6);
        System.out.println("Viewed as 1D tensor: " + viewed);
        
        Tensor autoDim = t.view(-1, 3);
        System.out.println("Auto dimension inference (-1, 3):\n" + autoDim);
        
        Tensor expanded = t.reshape(2, 3, 1);
        System.out.println("Expanded to 3D:\n" + expanded);
        
        Tensor squeezed = expanded.squeeze(2);
        System.out.println("Squeezed dimension 2:\n" + squeezed);
        
        Tensor unsqueezed = t.unsqueeze(0);
        System.out.println("Unsqueezed at dimension 0:\n" + unsqueezed);
    }
    
    private static void testAutogradFeatures() {
        System.out.println("\n6. Autograd Features:");
        
        Tensor a = Tensor.matrix(2, 2, 1, 2, 3, 4).requiresGrad(true);
        Tensor b = Tensor.matrix(2, 2, 5, 6, 7, 8).requiresGrad(true);
        
        System.out.println("a (requires grad):\n" + a);
        System.out.println("b (requires grad):\n" + b);
        
        Tensor c = a.addWithGrad(b);
        System.out.println("c = a + b:\n" + c);
        
        Tensor d = c.mulWithGrad(a);
        System.out.println("d = c * a:\n" + d);
        
        double sum = d.sum();
        Tensor e = Tensor.of(new int[]{1, 1}, sum).requiresGrad(true);
        System.out.println("e = sum(d): " + e);
        
        System.out.println("Computing gradients...");
        Tensor grad = Tensor.ones(1, 1);
        d.backward(grad);
        
        System.out.println("Gradient of a:\n" + a.grad());
        System.out.println("Gradient of b:\n" + b.grad());
        System.out.println("Gradient of c:\n" + c.grad());
        System.out.println("Gradient of d:\n" + d.grad());
        
        System.out.println("\nComparing with manual gradients:");
        
        Tensor a2 = Tensor.matrix(2, 2, 1, 2, 3, 4);
        Tensor b2 = Tensor.matrix(2, 2, 5, 6, 7, 8);
        
        Tensor c2 = a2.plus(b2);
        Tensor d2 = c2.times(a2);
        
        Tensor gradD2 = Tensor.ones(d2.shape());
        Tensor gradC2 = gradD2.times(a2);
        Tensor gradA2 = gradD2.times(c2).plus(gradC2);
        Tensor gradB2 = gradC2;
        
        System.out.println("Manual grad of a:\n" + gradA2);
        System.out.println("Manual grad of b:\n" + gradB2);
    }
} 