package net.echo.brain4j.clustering;

import net.echo.math4j.math.vector.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An implementation of the K-Means clustering algorithm.
 */
public class KMeans {

    private final List<Cluster> clusters;
    private final int clustersAmount;

    /**
     * Constructs a KMeans instance with a specified amount of clusters.
     *
     * @param clusters the amount of clusters
     */
    public KMeans(int clusters) {
        this.clusters = new ArrayList<>();
        this.clustersAmount = clusters;
    }

    /**
     * Initializes the clusters with the specified dimension.
     *
     * @param dimension the dimension for each cluster, must be the same as the dimension of the data
     */
    public void init(int dimension) {
        for (int i = 0; i < clustersAmount; i++) {
            clusters.add(new Cluster(dimension, i));
        }
    }

    /**
     * Makes one step into the K-Means algorithm.
     *
     * @param data the data to cluster
     * @return true if the centroids changed, false otherwise
     */
    public boolean step(ClusterData data) {
        boolean centroidsChanged = false;

        for (Cluster cluster : clusters) {
            cluster.clearData();
        }

        for (Vector vector : data.getData()) {
            Cluster closestCluster = getClosest(vector);

            closestCluster.addVector(vector);
        }

        for (Cluster cluster : clusters) {
            boolean updated = cluster.updateCenter();

            if (updated) {
                centroidsChanged = true;
            }
        }

        return centroidsChanged;
    }

    /**
     * Fits the model repetitively to the given dataset using the K-Means clustering algorithm.
     *
     * @param data the dataset to cluster
     * @return the number of iterations performed
     */
    public int fit(ClusterData data, int maxIterations) {
        boolean centroidsChanged = true;

        int i = 0;

        while (centroidsChanged) {
            centroidsChanged = step(data);

            if (i++ > maxIterations) break;
        }

        return i;
    }

    /**
     * Evaluates the dataset and maps each data point to its corresponding cluster.
     *
     * @param set the dataset to evaluate
     * @return a map of data points to their closest clusters
     */
    public Map<Vector, Cluster> evaluate(ClusterData set) {
        Map<Vector, Cluster> clusterMap = new HashMap<>();

        for (Vector vector : set.getData()) {
            Cluster closestCluster = getClosest(vector);

            clusterMap.put(vector, closestCluster);
        }

        return clusterMap;
    }

    /**
     * Retrieves the cluster closest to the given data point.
     * @param point the data point
     * @return the closest cluster
     */
    public Cluster getClosest(Vector point) {
        double minDistance = Double.MAX_VALUE;
        Cluster closestCluster = null;

        for (Cluster cluster : clusters) {
            double distance = cluster.getCenter().distance(point);

            if (distance < minDistance) {
                minDistance = distance;
                closestCluster = cluster;
            }
        }

        if (closestCluster == null) {
            throw new RuntimeException("Could not find closest cluster.");
        }

        return closestCluster;
    }

    public List<Cluster> getClusters() {
        return clusters;
    }
}
