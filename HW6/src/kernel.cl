__kernel void convolution(__constant float *filter, 
    __global const float *image, __global float *output, 
    const int filterWidth, const int imageWidth,
    const int imageHeight){
    
    int id = get_global_id(0);
    int i = id / imageWidth;
    int j = id % imageWidth;
    int index = imageWidth * i + j;

    int halffilterSize = filterWidth / 2;
    float sum;
    int k, l;

    sum = 0; // Reset sum for new source pixel
    // Apply the filter to the neighborhood
    for (k = -halffilterSize; k <= halffilterSize; k++){
        for (l = -halffilterSize; l <= halffilterSize; l++){
            if (i + k >= 0 && i + k < imageHeight &&
                j + l >= 0 && j + l < imageWidth){
                sum += image[(i + k) * imageWidth + j + l] *
                filter[(k + halffilterSize) * filterWidth +
                l + halffilterSize];
            }
        }   
    }
            
    output[index] = sum;
     
}
