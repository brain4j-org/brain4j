package org.brain4j.core.layer.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer0;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

/**
 * Graph Convolutional Network (GCN) layer implementation.
 * <p>
 * This layer applies the graph convolution operation. It combines node features with
 * the normalized adjacency matrix to propagate information across graph neighborhoods.
 * </p>
 * <h2>Shape conventions:</h2>
 * <p>Input:</p>
 * <ul>
 *     <li>{@code features}: {@code [batch, nodes, in_features]}</li>
 *     <li>{@code adjacency}: {@code [batch, nodes, nodes]}</li>
 * </ul>
 * <p>Output:</p>
 * <ul>
 *     <li>{@code [batch, nodes, out_features]}</li>
 *     <li>the adjacency matrix is returned as second output for downstream layers</li>
 * </ul>
 * <p>The adjacency normalization step computes:</p>
 * <blockquote><pre>
 * Â = D^{-0.5} (A + I) D^{-0.5}
 * </pre></blockquote>
 * where {@code A} is the adjacency matrix, {@code I} is the identity matrix,
 * and {@code D} is the degree matrix of {@code A + I}.
 * </p>
 *
 * @implNote this layer expects two input tensors for it to work
 * @author xEcho1337
 */
public class GraphConvLayer extends Layer0 {

    private int dimension;

    private GraphConvLayer() {
    }
    
    /**
     * Creates a GCN layer with the given output dimension and activation function.
     *
     * @param dimension the output feature dimension per node
     * @param activation the activation function applied after convolution
     */
    public GraphConvLayer(int dimension, Activation activation) {
        this.dimension = dimension;
        this.activation = activation;
    }
    
    /**
     * Creates a GCN layer with the given output dimension and activation function enum.
     *
     * @param dimension  the output feature dimension per node
     * @param activation the activation type (wrapped into an {@link Activation})
     */
    public GraphConvLayer(int dimension, Activations activation) {
        this(dimension, activation.function());
    }
    
    /**
     * Creates a GCN layer with the given output dimension using a linear activation.
     * @param dimension the output feature dimension per node
     */
    public GraphConvLayer(int dimension) {
        this(dimension, Activations.LINEAR);
    }

    @Override
    public void connect() {
        this.weights = Tensors.zeros(previous.size(), dimension).withGrad();
        this.bias = Tensors.zeros(dimension).withGrad();
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        validateInputLength(inputs);

        Tensor features = inputs[0]; // [batch, nodes, features]
        Tensor adjacencyMatrix = inputs[1]; // [batch, nodes, nodes]
        
        validateInputTensor(features, "Features must have shape [batch, nodes, features]!");
        validateInputTensor(adjacencyMatrix, "Adjacency matrix must have shape [batch, nodes, nodes]!");
        
        // [batch, nodes, dimension]
        Tensor support = features.matmulGrad(weights);
        Tensor adjNorm = normalizeAdjacency(adjacencyMatrix);

        // [batch, nodes, dimension]
        Tensor out = adjNorm.matmulGrad(support)
            .addGrad(bias);

        cache.recordOutput(this, out);

        return new Tensor[] { out.activateGrad(activation), adjacencyMatrix };
    }
    
    @Override
    public int size() {
        return dimension;
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dimension", dimension);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.dimension = object.get("dimension").getAsInt();
    }
    
    @Override
    public boolean validInput(Tensor input) {
        return input.rank() == 3;
    }

    private Tensor normalizeAdjacency(Tensor adjacent) {
        int batch = adjacent.shapeAt(0);
        int nodes = adjacent.shapeAt(1);

        Tensor identity = Tensors.eye(nodes);
        Tensor Ahat = adjacent.add(identity); // [batch, nodes, nodes]
        Tensor deg = Ahat.sum(1, false).add(1e-12); // [batch, nodes]
        Tensor degInvSqrt = deg.pow(-0.5); // shortcut for 1 / sqrt(deg)
        Tensor DinvSqrt = Tensors.zeros(batch, nodes, nodes);

        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < nodes; i++) {
                double value = degInvSqrt.get(b, i);
                DinvSqrt.set(value, b, i, i);
            }
        }
        
        return DinvSqrt.matmulGrad(Ahat).matmulGrad(DinvSqrt);
    }
}
