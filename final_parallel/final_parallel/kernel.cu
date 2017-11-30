#define _GNU_SOURCE
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <omp.h>
#include <iostream>
#include <cuda.h>
#include <conio.h>

typedef struct
{
	float* coordinates;
} Point;

typedef struct
{
	Point center;
	Point* points;
	float radius;
	int numOfPoints;
} Cluster;

__global__ void calculateCenter(float* coordinates, const int NUM_OF_POINTS)
{
    int i = threadIdx.x;
	//printf("Before: %f\n", coordinates[i]);
    coordinates[i] = coordinates[i] / NUM_OF_POINTS;
	//printf("After: %f\n", coordinates[i]);
}

__global__ void calculateClusterDGlobal(float* distances, float* cords, const int NUM_OF_DIMENSIONS, int NUM_OF_POINTS, int *current_thread_count)
{
	printf("%d\n", blockIdx.y);
	
	int mainPointIndex = blockIdx.x;
	int secondaryPointIndex = threadIdx.x;
	printf("%d\n", current_thread_count[0]);	
	int current_val = *current_thread_count;
	for (int j = 0 ; j < NUM_OF_DIMENSIONS ; j++)
	{
		float c1 = cords[mainPointIndex*NUM_OF_DIMENSIONS + j];
		float c2 = cords[secondaryPointIndex*NUM_OF_DIMENSIONS + j];
		distances[current_val] = (c1 - c2)*(c1 - c2);
	}
	
	distances[current_val] = sqrt(distances[current_val]);
	/*printf("againstPoint*NUM_OF_DIMENSIONS = %d\n", whichPoint*NUM_OF_DIMENSIONS);
	printf("againstPoint*NUM_OF_DIMENSIONS = %d\n", againstPoint*NUM_OF_DIMENSIONS);*/
	printf("distances[%d] = %f\n", current_val, distances[current_val]);
	atomicAdd(current_thread_count, 1);
}

/*int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}*/

// Helper function for using CUDA to add vectors in parallel.
float* calculateCenterUsingCuda(float* coordinates, const int NUM_OF_DIMENSIONS, const int NUM_OF_POINTS)
{
    float* coordinates_dev = 0;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    cudaStatus = cudaMalloc((void**)&coordinates_dev, NUM_OF_DIMENSIONS * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMemcpy(coordinates_dev, coordinates, NUM_OF_DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    calculateCenter <<< 1, NUM_OF_DIMENSIONS >>>(coordinates_dev, NUM_OF_POINTS);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "calculateCenterlaunch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateCenter!\n", cudaStatus);
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(coordinates, coordinates_dev, NUM_OF_DIMENSIONS * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaFree(coordinates_dev);

	cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

    return coordinates;
}

__global__ void calculateDistancOnlyCord(float* points, float* sharedResults, int numOfPoints, int NUM_OF_DIMENSIONS)
{
	int col = threadIdx.x; // 52
	int row = blockIdx.x; //811
	int pos = row*NUM_OF_DIMENSIONS + col;
	int block = NUM_OF_DIMENSIONS*numOfPoints;

	//Calculate distance between point P1 and all other points (x1-x2)x(x1-x2), (y1-y2)X(y1-y2), ....... and put results in shared temp matrix

	for (int i = 0; i < numOfPoints; i++) {

		sharedResults[(block*row) + (i*NUM_OF_DIMENSIONS) + col] = (points[pos] - points[i*NUM_OF_DIMENSIONS + col])*(points[pos] - points[i*NUM_OF_DIMENSIONS + col]);
	}
};


__global__ void calculateDistanceBetweenAllPointsAndCords(float* sharedResults, int numOfPoints, int NUM_OF_DIMENSIONS, float * results)
{
	//1 block, 811 threads
	int tid = threadIdx.x;
	int block = NUM_OF_DIMENSIONS*numOfPoints;
	int startIdx = tid*block;
	//int endIdx = tid*block+block;

	float sum = 0;
	//Work on blocks of 811 rows - sum and calc square root
	for (int i = 0; i < numOfPoints; i++) {
		sum = 0;
		for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
			sum += sharedResults[startIdx + (i*NUM_OF_DIMENSIONS) + j];
		}

		results[tid*numOfPoints + i] = sqrt(sum);
	}

};

float* calculateClusterD(Cluster* cluster, const int NUM_OF_DIMENSIONS)
{

	float* temp_cords = (float*)calloc(cluster->numOfPoints  * NUM_OF_DIMENSIONS, sizeof(float));
	cudaError_t cudaStatus;
	int count = 0;
	for(int i = 0 ; i < cluster->numOfPoints ; i++)
	{
		for(int j = 0 ; j < NUM_OF_DIMENSIONS ; j++)
		{
			temp_cords[count] = cluster->points[i].coordinates[j];
			count++;
		}
	}
	
	float* dev_points;
	float* sharedResults;
	float* results;
	float * resultsFromCuda = new float[cluster->numOfPoints*cluster->numOfPoints];
	

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//error(dev_points, sharedResults, results);
	}

	// Allocate GPU buffers for Array of all points
	cudaStatus = cudaMalloc((void**)&dev_points, sizeof(float)*NUM_OF_DIMENSIONS*cluster->numOfPoints);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	// Allocate GPU buffers for Results array
	cudaStatus = cudaMalloc((void**)&results, sizeof(float)*cluster->numOfPoints*cluster->numOfPoints);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	// Allocate GPU buffers for Shared results array
	cudaStatus = cudaMalloc((void**)&sharedResults, sizeof(float)*NUM_OF_DIMENSIONS*cluster->numOfPoints*cluster->numOfPoints);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}


	// Copy input array of points from host memory to GPU buffers.
	cudaStatus = cudaMemcpyAsync(dev_points, temp_cords, sizeof(float)*NUM_OF_DIMENSIONS*cluster->numOfPoints, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_coordinates_1 failed!");
	}



	// Launch a kernel on the GPU to calculate partial distances by coordinate .
	calculateDistancOnlyCord << <cluster->numOfPoints, NUM_OF_DIMENSIONS >> >(dev_points, sharedResults, cluster->numOfPoints, NUM_OF_DIMENSIONS);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calcExtendHistograma launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcExtendHistograma!\n", cudaStatus);
	}

	// Launch a kernel on the GPU to calculate distances between each 2 points
	calculateDistanceBetweenAllPointsAndCords << <1, cluster->numOfPoints >> >(sharedResults, cluster->numOfPoints, NUM_OF_DIMENSIONS, results);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calcExtendHistograma launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcExtendHistograma!\n", cudaStatus);
	}


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpyAsync(resultsFromCuda, results, sizeof(float)*cluster->numOfPoints*cluster->numOfPoints, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	cudaFree(dev_points);
	cudaFree(sharedResults);
	cudaFree(results);

	return resultsFromCuda;

}

/*cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}*/
