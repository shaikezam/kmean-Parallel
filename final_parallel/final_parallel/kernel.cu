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
	printf("distances[%d] = %f\n", current_val, distances[current_val]);
	atomicAdd(current_thread_count, 1);
}

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

/*
Main concept:
1. Get all cords of all points in cluster in array of floats.
2. Calculate all (Yn - -Ym)*(Yn - Ym) for all cords and put it in the temp array from section (1) [calculateDistancOnlyCord]
3. Iterate over all point in the temp array and sum it into the results array [calculateDistanceBetweenAllPointsAndCords]
4. Using OMP find the max value in the results array --> this is the max diameter
*/

__global__ void calculateDistancOnlyCord(float* points, float* allCordsInCluster, int numOfPoints, int NUM_OF_DIMENSIONS)
{
	int cord = threadIdx.x; 
	int pointIndex = blockIdx.x; 
	int pointTotalCords = pointIndex*NUM_OF_DIMENSIONS + cord;
	int totalCords = NUM_OF_DIMENSIONS*numOfPoints;

	//threadIdx.x = 0	blockIdx.x = 0						threadIdx.x = 1		blockIdx.x = 0
	//allCordsInCluster[0] = (P0C0 - P0C0)*(P0C0 - P0C0)	allCordsInCluster[1] = (P0C1 - P0C1)*(P0C1 - P0C1)
	//allCordsInCluster[52] = (P0C0 - P1C0)*(P0C0 - P1C0)	allCordsInCluster[53] = (P0C1 - P1C1)*(P0C0 - P1C1)
	//allCordsInCluster[104] = (P0C0 - P2C0)*(P0C0 - P2C0)	allCordsInCluster[105] = (P0C1 - P2C1)*(P0C1 - P2C1)
	for (int i = 0; i < numOfPoints; i++) {
		allCordsInCluster[(totalCords*pointIndex) + (i*NUM_OF_DIMENSIONS) + cord] = (points[pointTotalCords] - points[i*NUM_OF_DIMENSIONS + cord])*(points[pointTotalCords] - points[i*NUM_OF_DIMENSIONS + cord]);
	}
};


__global__ void calculateDistanceBetweenAllPointsAndCords(float* allCordsInCluster, int numOfPoints, int NUM_OF_DIMENSIONS, float * results)
{

	int pointIndex = threadIdx.x;
	int totalCords = NUM_OF_DIMENSIONS*numOfPoints;
	int startIdx = pointIndex*totalCords;


	float sum = 0;

	for (int i = 0; i < numOfPoints; i++) {
		sum = 0;
		for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
			sum += allCordsInCluster[startIdx + (i*NUM_OF_DIMENSIONS) + j];
		}

		results[pointIndex*numOfPoints + i] = sqrt(sum);
	}

};


float* calculateClusterD(Cluster* cluster, const int NUM_OF_DIMENSIONS)
{

	float* temp_cords = (float*)calloc(cluster->numOfPoints  * NUM_OF_DIMENSIONS, sizeof(float));
	cudaError_t cudaStatus;
	int count = 0;
	//get all cords to single floats array
	for(int i = 0 ; i < cluster->numOfPoints ; i++)
	{
		for(int j = 0 ; j < NUM_OF_DIMENSIONS ; j++)
		{
			temp_cords[count] = cluster->points[i].coordinates[j];
			count++;
		}
	}
	
	float* dev_points;
	float* allCordsInCluster;
	float* results;
	//array of all points's distances of all points in clusters
	float * resultsFromCuda = new float[cluster->numOfPoints*cluster->numOfPoints];
	

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	// Allocate array of all cords of all points in clusters
	cudaStatus = cudaMalloc((void**)&dev_points, sizeof(float)*NUM_OF_DIMENSIONS*cluster->numOfPoints);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	// Allocate array of all points's distances of all points in clusters
	cudaStatus = cudaMalloc((void**)&results, sizeof(float)*cluster->numOfPoints*cluster->numOfPoints);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	// Allocate temp array of all points's distances of all points in clusters
	cudaStatus = cudaMalloc((void**)&allCordsInCluster, sizeof(float)*NUM_OF_DIMENSIONS*cluster->numOfPoints*cluster->numOfPoints);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}


	// Copy input array of points from host memory to GPU buffers.
	cudaStatus = cudaMemcpyAsync(dev_points, temp_cords, sizeof(float)*NUM_OF_DIMENSIONS*cluster->numOfPoints, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_coordinates_1 failed!");
	}



	// Launch a kernel on the GPU to calculate partial distances by coordinate .
	calculateDistancOnlyCord << <cluster->numOfPoints, NUM_OF_DIMENSIONS >> >(dev_points, allCordsInCluster, cluster->numOfPoints, NUM_OF_DIMENSIONS);


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
	calculateDistanceBetweenAllPointsAndCords << <1, cluster->numOfPoints >> >(allCordsInCluster, cluster->numOfPoints, NUM_OF_DIMENSIONS, results);


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
	cudaFree(allCordsInCluster);
	cudaFree(results);

	return resultsFromCuda;

}