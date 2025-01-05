package net.echo.brain4j.clustering;

import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KMeans {

    private final List<Cluster> clusters;
    private final int clustersSize;

    /**
     * Constructs a new KMeans object with a specified amount of clusters.
     *
     * @param clustersSize the amount of clusters
     */
    public KMeans(int clustersSize) {
        this.clusters = new ArrayList<>();
        this.clustersSize = clustersSize;
    }

    /**
     * Initializes the clusters with the specified dimension.
     *
     * @param dimension the dimension for each cluster, must be the same as the dimension of the data
     */
    public void init(int dimension) {
        for (int i = 0; i < clustersSize; i++) {
            clusters.add(new Cluster(dimension, i));
        }
    }

    /**
     * Fits the model to the given dataset using the K-Means clustering algorithm.
     *
     * @param set the dataset to cluster
     */
    public void fit(ClusterData set, int maxIterations) {
        boolean centroidsChanged = true;

        int i = 0;

        while (centroidsChanged) {
            centroidsChanged = false;

            for (Vector vector : set.getData()) {
                Cluster closestCluster = getClosest(vector);

                closestCluster.addVector(vector);
            }

            for (Cluster cluster : clusters) {
                boolean updated = cluster.updateCenter();

                if (updated) {
                    centroidsChanged = true;
                }
            }

            for (Cluster cluster : clusters) {
                cluster.clearData();
            }

            if (i++ > maxIterations) break;
        }
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

    private Cluster getClosest(Vector point) {
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
