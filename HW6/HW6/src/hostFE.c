#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#define MAX_SOURCE_SIZE (0x100000)

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;
    

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &status);

    // Create memory buffers on the device for each vector 
    cl_mem cl_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, 
            filterSize * sizeof(float), NULL, &status);
    cl_mem cl_image = clCreateBuffer(*context, CL_MEM_READ_ONLY,
            imageSize * sizeof(float), NULL, &status);
    cl_mem cl_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, 
            imageSize * sizeof(float), NULL, &status);

    
    // Copy the lists A and B to their respective memory buffers
    status = clEnqueueWriteBuffer(command_queue, cl_filter, CL_TRUE, 0,
            filterSize * sizeof(float), filter, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(command_queue, cl_image, CL_TRUE, 0, 
            imageSize * sizeof(float), inputImage, 0, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    // Set the arguments of the kernel
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cl_filter);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cl_image);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&cl_output);
    status = clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&filterWidth);
    status = clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&imageWidth);
    status = clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&imageHeight);

    
    // Execute the OpenCL kernel on the list
    size_t global_item_size = imageSize; // Process the entire lists
    size_t local_item_size = 64; // Divide work items into groups of 64
    status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL);
    
    // Read the memory buffer C on the device to the local variable C
    status = clEnqueueReadBuffer(command_queue, cl_output, CL_TRUE, 0, 
            imageSize * sizeof(float), outputImage, 0, NULL, NULL);


     // Clean up
    status = clFlush(command_queue);
    status = clFinish(command_queue);
    status = clReleaseKernel(kernel);
//     status = clReleaseProgram(program);
    status = clReleaseMemObject(cl_filter);
    status = clReleaseMemObject(cl_image);
    status = clReleaseMemObject(cl_output);
    status = clReleaseCommandQueue(command_queue);
//     status = clReleaseContext(context);
    return 0;

}