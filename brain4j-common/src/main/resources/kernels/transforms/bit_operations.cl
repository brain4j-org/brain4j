#include "../common_definitions.cl"

inline int bit_reverse(int x, int bits) {
    int r = 0;
    for (int i = 0; i < bits; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
} 