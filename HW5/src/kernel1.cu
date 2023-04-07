#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY, int count, int* Md, int resX) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float c_re = lowerX + thisX * stepX;
    float c_im = lowerY + thisY * stepY;

    float z_re = c_re;
    float z_im = c_im;
    int i;

    for (i=0; i<count; ++i){
        if (z_re * z_re + z_im * z_im > 4.f)
          break;
        
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    int index = thisY * resX + thisX;
    Md[index] = i;
    
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    dim3 t_per_b(16, 16);
    dim3 num_block(resX / t_per_b.x, resY / t_per_b.y);


    int *Md, *host_mem;
    int size = resX * resY * sizeof(int);
    cudaMalloc((void**) &Md, size);
    host_mem = (int *)malloc(size);
    
    mandelKernel<<<num_block, t_per_b>>>(stepX, stepY, lowerX, lowerY, maxIterations, Md, resX);
    cudaMemcpy(host_mem, Md, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img, host_mem, size, cudaMemcpyHostToHost);
    cudaFree(Md);
    free(host_mem);

}
