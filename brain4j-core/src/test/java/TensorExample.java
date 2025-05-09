import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

public class TensorExample {

    public static void main(String[] args) {
        Tensor a = Tensors.of(new int[]{2, 3}, 1, 2, 3, 4, 5, 6);

        System.out.println(a);

        Tensor t = a.transpose();

        System.out.println("Transpose");
        System.out.println(t);
    }
}
