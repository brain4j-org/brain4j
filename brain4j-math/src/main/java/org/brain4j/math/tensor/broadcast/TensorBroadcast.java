package org.brain4j.math.tensor.broadcast;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.broadcast.impl.*;

public class TensorBroadcast {

    public static Tensor add(Tensor A, Tensor B) {
        return new BroadcastAdd().defaultOp(A, B);
    }

    public static Tensor sub(Tensor A, Tensor B) {
        return new BroadcastSub().defaultOp(A, B);
    }

    public static Tensor mul(Tensor A, Tensor B) {
        return new BroadcastMul().defaultOp(A, B);
    }

    public static Tensor div(Tensor A, Tensor B) {
        return new BroadcastDiv().defaultOp(A, B);
    }

    public static Tensor pow(Tensor A, Tensor B) {
        return new BroadcastPow().defaultOp(A, B);
    }

    public static Tensor forward(BroadcastOperation operation, Tensor A, Tensor B) {
        return operation.defaultOp(A, B);
    }
}
