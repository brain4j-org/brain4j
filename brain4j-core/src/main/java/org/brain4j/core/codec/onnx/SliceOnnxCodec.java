package org.brain4j.core.codec.onnx;

import org.brain4j.core.codec.OnnxCodec;
import org.brain4j.core.importing.onnx.ProtoOnnx.AttributeProto;
import org.brain4j.core.importing.onnx.ProtoOnnx.NodeProto;
import org.brain4j.math.commons.Range;
import org.brain4j.math.tensor.autograd.impl.SliceOperation;

import java.util.ArrayList;
import java.util.List;

public class SliceOnnxCodec implements OnnxCodec<SliceOperation> {

    @Override
    public String type() {
        return "Slice";
    }

    @Override
    public Class<SliceOperation> targetClass() {
        return SliceOperation.class;
    }

    @Override
    public void encode(SliceOperation op, NodeProto.Builder builder) {
        Range[] ranges = op.ranges();

        for (int dim = 0; dim < ranges.length; dim++) {
            Range range = ranges[dim];

            builder.addAttribute(AttributeProto.newBuilder()
                .setName("starts")
                .setI(range.start())
                .setType(AttributeProto.AttributeType.INTS)
                .build());
            builder.addAttribute(AttributeProto.newBuilder()
                .setName("ends")
                .setI(range.end())
                .setType(AttributeProto.AttributeType.INTS)
                .build());
            builder.addAttribute(AttributeProto.newBuilder()
                .setName("axes")
                .setI(dim)
                .setType(AttributeProto.AttributeType.INTS)
                .build());
            builder.addAttribute(AttributeProto.newBuilder()
                .setName("steps")
                .setI(range.step())
                .setType(AttributeProto.AttributeType.INTS)
                .build());
        }
    }

    @Override
    public SliceOperation decode(NodeProto node) {
        List<Integer> starts = ints(node, "starts");
        List<Integer> ends = ints(node, "ends");
        List<Integer> axes = ints(node, "axes");
        List<Integer> steps = ints(node, "steps");

        if (starts.isEmpty()) {
            return new SliceOperation();
        }

        int rank = ends.isEmpty() ? starts.size() : Math.max(starts.size(), ends.size());

        if (axes.isEmpty()) {
            for (int d = 0; d < rank; d++) axes.add(d);
        }
        if (steps.isEmpty()) {
            for (int d = 0; d < rank; d++) steps.add(1);
        }

        Range[] ranges = new Range[rank];

        for (int d = 0; d < rank; d++) {
            int axis = d < axes.size() ? axes.get(d) : d;
            int start = d < starts.size() ? starts.get(d) : 0;
            int end = d < ends.size() ? ends.get(d) : Integer.MAX_VALUE;
            int step = d < steps.size() ? steps.get(d) : 1;

            ranges[axis] = new Range(start, end, step);
        }

        return new SliceOperation(ranges);
    }

    private static List<Integer> ints(NodeProto node, String name) {
        List<Integer> values = new ArrayList<>();
        for (AttributeProto attr : node.getAttributeList()) {
            if (name.equals(attr.getName())) values.add((int) attr.getI());
        }
        return values;
    }
}
