package org.brain4j.math.commons;

import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;

public class Batch extends Pair<Tensor[], Tensor[]> {

    public Batch(Tensor[] key, Tensor[] second) {
        super(key, second);
    }
    
    public Batch to(Device device) {
        Tensor[] newInputs = new Tensor[first.length];
        Tensor[] newLabels = new Tensor[second.length];
        
        for (int i = 0; i < newInputs.length; i++) {
            newInputs[i] = first[i].to(device);
        }
        
        for (int i = 0; i < newLabels.length; i++) {
            newLabels[i] = second[i].to(device);
        }
        
        return new Batch(newInputs, newLabels);
    }
}
