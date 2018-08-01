__device__ float clamp(float x, float a, float b) {
    return min(max(x, a), b);
}
