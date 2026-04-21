package org.brain4j.core.layer.old.impl.utility;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.brain4j.core.layer.old.OldLayer;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.commons.Range;

public class SliceLayer extends OldLayer {
    
    private Range[] ranges;
    private int size;

    protected SliceLayer() {
    }

    public SliceLayer(Range... ranges) {
        this.ranges = ranges;
    }
    
    @Override
    public void connect() {
        this.size = previous.size();
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] result = new Tensor[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            result[i] = inputs[i].sliceGrad(ranges);
        }

        return result;
    }
    
    @Override
    public int size() {
        return size;
    }
    
    @Override
    public void serialize(JsonObject object) {
        JsonArray rangesArray = new JsonArray();
        
        for (Range range : ranges) {
            JsonArray array = new JsonArray();
            array.add(range.start());
            array.add(range.end());
            array.add(range.step());
            rangesArray.add(array);
        }
        
        object.add("ranges", rangesArray);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        JsonArray rangesArray = object.get("ranges").getAsJsonArray();

        this.ranges = new Range[rangesArray.size()];
        
        for (int i = 0; i < ranges.length; i++) {
            JsonArray range = rangesArray.get(i).getAsJsonArray();
            int start = range.get(0).getAsInt();
            int end = range.get(1).getAsInt();
            int step = range.get(2).getAsInt();
            this.ranges[i] = new Range(start, end, step);
        }
    }
}
