package org.brain4j.core.clustering;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.ArrayList;
import java.util.List;

public class Cluster {

    private Tensor center;
    private final List<Tensor> vectors;
    private final int id;

    /**
     * Constructs a cluster with a centroid initialized randomly.
     *
     * @param dimension the dimension of the space (number of features)
     */
    public Cluster(int dimension, int id) {
        this.id = id;
        this.center = Tensors.random(dimension);
        this.vectors = new ArrayList<>();
    }

    /**
     * Returns the centroid of the cluster.
     *
     * @return the centroid as a Vector
     */
    public Tensor getCenter() {
        return center;
    }

    /**
     * Sets a new centroid for the cluster.
     *
     * @param center the new centroid
     */
    public void setCenter(Tensor center) {
        this.center = center;
    }

    /**
     * Adds a data point to the cluster.
     *
     * @param vector the data point to add
     */
    public void addVector(Tensor vector) {
        vectors.add(vector);
    }

    /**
     * Returns the data points assigned to the cluster.
     *
     * @return the list of data points
     */
    public List<Tensor> getVectors() {
        return vectors;
    }

    /**
     * Updates the centroid of the cluster by calculating the mean of the assigned data points.
     *
     * @return true if the centroid changed, false otherwise
     */
    public boolean updateCenter() {
        if (vectors.isEmpty()) {
            return false;
        }

        Tensor newCenter = Tensors.create(center.elements());

        for (Tensor row : vectors) {
            newCenter.add(row);
        }

        newCenter.divide(vectors.size());

        boolean isChanged = !center.equals(newCenter);
        this.center = newCenter;

        return isChanged;
    }

    /**
     * Clears all data points assigned to the cluster (used after an iteration of K-Means).
     */
    public void clearData() {
        vectors.clear();
    }

    /**
     * Retrieves the identifier of the cluster, as a number.
     *
     * @return the id of the cluster
     */
    public int getId() {
        return id;
    }
}
