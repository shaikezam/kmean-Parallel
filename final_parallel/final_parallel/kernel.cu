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

__global__ void calculateCenter2(Point* points, float* coordinates)
{
    int i = threadIdx.x;
	int y = blockIdx.x;
	//printf("y = %d\n", y);
	//printf("i = %d\n", i);
	if(i == 0)
	{
		printf("index = %d\n",i);
	}
	else
	{
		printf("index = %d\n",(y+i)*i);
	}
	Point point = points[y];
	//printf("Before: %f\n", coordinates[i]);
    coordinates[i*y + (i - y)] = point.coordinates[i];
	//printf("After: %f\n", coordinates[i]);
}

__global__ void calculateDistance(float* coordinates1, float* coordinates2, double* sum)
{
    int i = threadIdx.x;
	//printf("Before: %f\n", coordinates[i]);
    double temp = (coordinates1[i] - coordinates2[i])*(coordinates1[i] - coordinates2[i]);
	//atomicAdd(&sum[0], temp);
	//printf("After: %f\n", coordinates[i]);
}

__device__ int counter;

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

float* calculateCenterUsingCuda2(Cluster* cluster, const int NUM_OF_DIMENSIONS)
{
	const int numOfPoints = cluster->numOfPoints;
    float* coordinates_dev;
	float* coordinates = (float*)calloc(NUM_OF_DIMENSIONS * numOfPoints, sizeof(float));
	Point* points_dev;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    cudaStatus = cudaMalloc((void**)&coordinates_dev, NUM_OF_DIMENSIONS * numOfPoints * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

	cudaStatus = cudaMalloc((void**)&points_dev, numOfPoints * sizeof(Point));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMemcpy(coordinates_dev, coordinates, NUM_OF_DIMENSIONS * numOfPoints * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaStatus = cudaMemcpy(points_dev, cluster->points, numOfPoints * sizeof(Point), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    calculateCenter2 <<<numOfPoints, NUM_OF_DIMENSIONS>>>(points_dev, coordinates_dev);

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
	cudaFree(points_dev);

	cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

    return coordinates;
}

double* calculateDistanceBetween2PointsUsingCuda(float* coordinates1, float* coordinates2, double* sum, const int NUM_OF_DIMENSIONS)
{
	double* sum_dev = 0;
    float* coordinates_dev1 = 0;
	float* coordinates_dev2 = 0;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    cudaStatus = cudaMalloc((void**)&coordinates_dev1, NUM_OF_DIMENSIONS * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

	cudaStatus = cudaMalloc((void**)&coordinates_dev2, NUM_OF_DIMENSIONS * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

	cudaStatus = cudaMalloc((void**)&sum_dev, 1 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMemcpy(coordinates_dev1, coordinates1, NUM_OF_DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaStatus = cudaMemcpy(coordinates_dev2, coordinates2, NUM_OF_DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaStatus = cudaMemcpy(sum_dev, sum, 1 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    calculateDistance <<< 1, NUM_OF_DIMENSIONS >>>(coordinates_dev1, coordinates_dev2, sum);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "calculateCenterlaunch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateCenter!\n", cudaStatus);
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(coordinates1, coordinates_dev1, NUM_OF_DIMENSIONS * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaStatus = cudaMemcpy(coordinates2, coordinates_dev2, NUM_OF_DIMENSIONS * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaStatus = cudaMemcpy(sum, sum_dev, 1 * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaFree(coordinates_dev1);
	cudaFree(coordinates_dev2);
	cudaFree(sum_dev);

	cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

    return sum;
}

float* calculateClusterD(Cluster* cluster, const int NUM_OF_DIMENSIONS)
{
	int tally = 0, *dev_tally;

	cudaMalloc((void **)&dev_tally, sizeof(int));

	cudaMemcpy(dev_tally, &tally, sizeof(int), cudaMemcpyHostToDevice);

	float* temp_cords = (float*)calloc(cluster->numOfPoints  * NUM_OF_DIMENSIONS, sizeof(float));
	float* temp_cords_dev;
	int count = 0;
	for(int i = 0 ; i < cluster->numOfPoints ; i++)
	{
		for(int j = 0 ; j < NUM_OF_DIMENSIONS ; j++)
		{
			temp_cords[count] = cluster->points[i].coordinates[j];
			count++;
		}
	}
	float* temp_distances = (float*)calloc(cluster->numOfPoints  * cluster->numOfPoints * NUM_OF_DIMENSIONS, sizeof(float));
	float* temp_distances_dev;
	cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

	cudaStatus = cudaMalloc((void**)&temp_cords_dev, cluster->numOfPoints  * NUM_OF_DIMENSIONS * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

	cudaStatus = cudaMalloc((void**)&temp_distances_dev, cluster->numOfPoints  * cluster->numOfPoints * NUM_OF_DIMENSIONS * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

	cudaStatus = cudaMemcpy(temp_cords_dev, temp_cords, cluster->numOfPoints  * NUM_OF_DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaStatus = cudaMemcpy(temp_distances_dev, temp_distances, cluster->numOfPoints * cluster->numOfPoints * NUM_OF_DIMENSIONS * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	calculateClusterDGlobal <<< cluster->numOfPoints, cluster->numOfPoints >>>(temp_distances_dev, temp_cords_dev, NUM_OF_DIMENSIONS, cluster->numOfPoints, dev_tally);

	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "calculateCenterlaunch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateCenter!\n", cudaStatus);
    }

	// Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(temp_distances, temp_distances_dev, cluster->numOfPoints * cluster->numOfPoints * NUM_OF_DIMENSIONS * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaMemcpy(&tally, dev_tally, sizeof(int), cudaMemcpyDeviceToHost); 
	printf("total number of threads that executed was: %d\n", tally);

	//cudaFree(temp_distances_dev);
	//cudaFree(temp_cords_dev);
	//free(temp_cords);
	return temp_distances;
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
