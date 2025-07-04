package org.brain4j.core.importing.onnx;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

import java.util.List;

public record CachedNode(Operation operation, List<String> inputs, List<String> outputs, List<Tensor> weights) {
}
