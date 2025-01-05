import net.echo.brain4j.clustering.Cluster;
import net.echo.brain4j.clustering.ClusterData;
import net.echo.brain4j.clustering.KMeans;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class KMeansExample {

    public static void main(String[] args) {
        int numClusters = 2;
        int dimension = 2;

        List<Vector> dataRows = new ArrayList<>();
        dataRows.add(Vector.of(1.0, 1.0)); // Cluster 0
        dataRows.add(Vector.of(1.5, 1.5)); // Cluster 0
        dataRows.add(Vector.of(5.0, 5.0)); // Cluster 1
        dataRows.add(Vector.of(6.0, 6.0)); // Cluster 1
        dataRows.add(Vector.of(0.5, 0.5)); // Cluster 0
        dataRows.add(Vector.of(5.5, 5.5)); // Cluster 1

        ClusterData dataSet = new ClusterData(dataRows);

        KMeans kMeans = new KMeans(numClusters);
        kMeans.init(dimension);

        System.out.println("Training K-Means...");
        kMeans.fit(dataSet, 1);

        System.out.println("\nCluster Assignments:");
        var clusterAssignments = kMeans.evaluate(dataSet);

        for (Vector vector : dataRows) {
            Cluster cluster = clusterAssignments.get(vector);
            System.out.printf("Point %s -> Cluster (%d) center: %s%n", vector, cluster.getId(), cluster.getCenter());
        }

        System.out.println("\nFinal Cluster Centers:");

        for (Cluster cluster : kMeans.getClusters()) {
            System.out.println("Cluster center: " + cluster.getCenter());
        }
    }
}
