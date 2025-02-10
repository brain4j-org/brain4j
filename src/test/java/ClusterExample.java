import net.echo.brain4j.clustering.Cluster;
import net.echo.brain4j.clustering.ClusterData;
import net.echo.brain4j.clustering.KMeans;
import net.echo.brain4j.utils.Vector;

import java.util.Map;

public class ClusterExample {

    public static void main(String[] args) {
        ClusterExample example = new ClusterExample();
        example.start();
    }

    private void start() {
        ClusterData data = getData();

        KMeans kMeans = new KMeans(2); // 2 clusters
        kMeans.init(2); // bi-dimensional input
        kMeans.fit(data, 100);

        System.out.println("Cluster Assignments:");
        Map<Vector, Cluster> clusterAssignments = kMeans.evaluate(data);

        for (Vector vector : data) {
            Cluster cluster = clusterAssignments.get(vector);
            System.out.printf("Point %s -> Cluster (%d) center: %s%n", vector, cluster.getId(), cluster.getCenter());
        }

        System.out.println("Final Cluster Centers:");

        for (Cluster cluster : kMeans.getClusters()) {
            System.out.println("Cluster center: " + cluster.getCenter());
        }
    }

    private ClusterData getData() {
        ClusterData data = new ClusterData();

        data.add(Vector.of(1.0, 1.0)); // Cluster 0
        data.add(Vector.of(1.5, 1.5)); // Cluster 0
        data.add(Vector.of(0.5, 0.5)); // Cluster 0

        data.add(Vector.of(5.0, 5.0)); // Cluster 1
        data.add(Vector.of(6.0, 6.0)); // Cluster 1
        data.add(Vector.of(5.5, 5.5)); // Cluster 1

        return data;
    }
}
