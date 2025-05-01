package org.brain4j.core.adapters;

import java.io.DataInputStream;
import java.io.DataOutputStream;

public interface Adapter {

    void serialize(DataOutputStream stream) throws Exception;

    void deserialize(DataInputStream stream) throws Exception;
}
