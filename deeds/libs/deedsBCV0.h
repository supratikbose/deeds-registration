void deeds(float *im1, float *im1b, float *warped1, 
    float *flow_flattened_out,
    float *flow_W_out, float *flow_V_out, float *flow_U_out,
    int *depth_out, int *height_out, int *width_out, 
    int depth_in, int height_in, int width_in,
    bool defVectorResampledToVolume_in, 
    float alpha, int levels, bool verbose);
