package org.brain4j.math.broadcast;

import org.brain4j.math.broadcast.impl.*;
import org.brain4j.math.tensor.Tensor;

public class TensorBroadcast {

    private final static BroadcastAdd ADD_OP = new BroadcastAdd();
    private final static BroadcastSub SUB_OP = new BroadcastSub();
    private final static BroadcastMul MUL_OP = new BroadcastMul();
    private final static BroadcastDiv DIV_OP = new BroadcastDiv();
    private final static BroadcastPow POW_OP = new BroadcastPow();
    
    public static Tensor add(Tensor A, Tensor B) {
        return ADD_OP.defaultOp(A, B);
    }

    public static Tensor sub(Tensor A, Tensor B) {
        return SUB_OP.defaultOp(A, B);
    }

    public static Tensor mul(Tensor A, Tensor B) {
        return MUL_OP.defaultOp(A, B);
    }

    public static Tensor div(Tensor A, Tensor B) {
        return DIV_OP.defaultOp(A, B);
    }

    public static Tensor pow(Tensor A, Tensor B) {
        return POW_OP.defaultOp(A, B);
    }

    public static Tensor forward(BroadcastOperation operation, Tensor A, Tensor B) {
        return operation.defaultOp(A, B);
    }
}
