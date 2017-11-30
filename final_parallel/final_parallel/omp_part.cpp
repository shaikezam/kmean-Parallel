#define _GNU_SOURCE
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "myHeader.h"
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <omp.h>
#include <iostream>
#include <conio.h>
#include <time.h>

int main(int argc,char *argv[])
{
	clock_t start, finish;  
    start = clock();  
	int NUM_OF_DIMENSIONS, NUM_OF_PRODUCTS, MAX_NUM_OF_CLUSTERS, MAX_NUM_OF_ITERATION;
	float QM;
	Cluster* clusters = (Cluster*)calloc(NUM_OF_CLUSTERS, sizeof(Cluster));
	//read points from file
	Point* points = readDataFromFile(&NUM_OF_DIMENSIONS, &NUM_OF_PRODUCTS, &MAX_NUM_OF_CLUSTERS, &MAX_NUM_OF_ITERATION, &QM, clusters, NUM_OF_CLUSTERS);	
	double tempQM;
	for( ; NUM_OF_CLUSTERS < MAX_NUM_OF_CLUSTERS ; )
	{
		//calculate cluster centers
		checkIsCurrentClustersEnough(points, clusters, NUM_OF_DIMENSIONS, NUM_OF_PRODUCTS, NUM_OF_CLUSTERS, MAX_NUM_OF_ITERATION);
		tempQM = calculateQM(clusters, NUM_OF_DIMENSIONS, NUM_OF_CLUSTERS);
		printf("With %d clusters the QM is: %f\n", NUM_OF_CLUSTERS, tempQM);
		if(tempQM <= QM)
		{
			printf("The QM is: %f, GoodBye\n",tempQM);
			finish = clock();
			printf("Program took %f\n", ((float)(finish - start) / 1000000.0F ) * 1000);
			printOutputFile(clusters, NUM_OF_CLUSTERS, NUM_OF_DIMENSIONS, tempQM);
			exit(1);
		}
		NUM_OF_CLUSTERS++;
		free(clusters);
		free(points);
		clusters = (Cluster*)calloc(NUM_OF_CLUSTERS, sizeof(Cluster));
		points = readDataFromFile(&NUM_OF_DIMENSIONS, &NUM_OF_PRODUCTS, &MAX_NUM_OF_CLUSTERS, &MAX_NUM_OF_ITERATION, &QM, clusters, NUM_OF_CLUSTERS);	
		//clusters = appendPointsAsClusters(clusters, points, &NUM_OF_DIMENSIONS, NUM_OF_CLUSTERS);
	}
	finish = clock();
	printf("Program took %f\n", ((float)(finish - start) / 1000000.0F ) * 1000);
	printOutputFile(clusters, NUM_OF_CLUSTERS, NUM_OF_DIMENSIONS, tempQM);
}

Point *readDataFromFile(int* NUM_OF_DIMENSIONS, int* NUM_OF_PRODUCTS, int* MAX_NUM_OF_CLUSTERS, int* MAX_NUM_OF_ITERATION, float* QM, Cluster* clusters, const int NUM_OF_CLUSTERS)
{
	FILE* file = fopen(pathToFile, "r");
	int rc = fscanf(file, "%d,%d,%d,%d,%f\n", NUM_OF_PRODUCTS, NUM_OF_DIMENSIONS, MAX_NUM_OF_CLUSTERS, MAX_NUM_OF_ITERATION, QM);
	Point* p = (Point*)calloc(*NUM_OF_PRODUCTS, sizeof(Point));

	for(int i = 0 ; i < *NUM_OF_PRODUCTS ; i++)
	{
		p[i].coordinates = (float*)calloc(*NUM_OF_DIMENSIONS, sizeof(float));
		float temp = 0;
		for(int j = 0 ; j < *NUM_OF_DIMENSIONS ; j++)
		{
			if (j == *NUM_OF_DIMENSIONS-1) 
			{
				int rc = fscanf(file, "%f\n", &temp);
			}
			else 
			{
				int rc = fscanf(file, "%f,", &temp);
			}
			p[i].coordinates[j] = temp;
		}

	}
	for(int i = 0 ; i < NUM_OF_CLUSTERS ; i++)
	{
		//clusters[i].center.coordinates = (float*)calloc(*NUM_OF_DIMENSIONS, sizeof(float));
		clusters[i].center = p[i];
		clusters[i].numOfPoints = 0;
		clusters[i].radius = 0;
	}
	return p;
}

void addPointsToClusters(Point* points, Cluster* clusters, const int NUM_OF_PRODUCTS, const int NUM_OF_CLUSTERS, const int NUM_OF_DIMENSIONS)
{
	for (int i = 0 ; i < NUM_OF_PRODUCTS ; i++)
	{
		int clusterIndex = 0;
		double smallestValue = 1000000;
		for (int j = 0; j < NUM_OF_CLUSTERS ; j++)
		{
			double tempRangValue = getDistanceBetweenTwoPoints(points[i], clusters[j].center, NUM_OF_DIMENSIONS);
			if(smallestValue >= tempRangValue)
			{
				smallestValue = tempRangValue;
				clusterIndex = j;
			}
		}
		clusters[clusterIndex].numOfPoints++;
	}
	for(int i = 0 ; i < NUM_OF_CLUSTERS ; i++)
	{
		//printf("Num of points %d\n", clusters[i].numOfPoints);
		clusters[i].points = (Point *) calloc(clusters[i].numOfPoints, sizeof(Point));
		clusters[i].numOfPoints = 0;
	}
	for (int i = 0 ; i < NUM_OF_PRODUCTS ; i++)
	{
		int clusterIndex = 0;
		double smallestValue = 10000;
		for (int j = 0; j < NUM_OF_CLUSTERS ; j++)
		{
			double tempRangValue = getDistance(points[i], clusters[j], NUM_OF_DIMENSIONS);
			if(smallestValue >= tempRangValue)
			{
				smallestValue = tempRangValue;
				clusterIndex = j;
			}	
		}
		int index = clusters[clusterIndex].numOfPoints++;
		clusters[clusterIndex].points[index] = points[i];
	}
}

Cluster* appendPointsAsClusters(Cluster* clusters, Point* points, int* NUM_OF_DIMENSIONS, const int NUM_OF_CLUSTERS)
{
	Cluster* clusters1 = (Cluster*)calloc(NUM_OF_CLUSTERS, sizeof(Cluster));
	if(clusters1 == NULL)
	{
		clusters1 = (Cluster*)calloc(NUM_OF_CLUSTERS, sizeof(Cluster));
	}
	for(int i = 0 ; i < NUM_OF_CLUSTERS ; i++)
	{
		clusters1[i].center = points[i];
		clusters1[i].numOfPoints = 0;
		//clusters[i].points = NULL;
		clusters1[i].radius = 0;
	}
	return clusters1;
}

double getDistance(Point point, Cluster cluster, const int NUM_OF_DIMENSIONS)
{
	double sum = 0;
	for(int i = 0 ; i < NUM_OF_DIMENSIONS ; i++)
	{
		sum += (point.coordinates[i] - cluster.center.coordinates[i]) * (point.coordinates[i] - cluster.center.coordinates[i]);

	}
	return sqrt(sum);
}

double getDistanceBetweenTwoPoints(Point point1, Point point2, const int NUM_OF_DIMENSIONS)
{
	double sum = 0;
	for(int i = 0 ; i < NUM_OF_DIMENSIONS ; i++)
	{
		sum += (point1.coordinates[i] - point2.coordinates[i])*(point1.coordinates[i] - point2.coordinates[i]);
	}
	return sqrt(sum);
}

void removePointFromCluster(Cluster* clusters, const int NUM_OF_CLUSTERS)
{
	for(int i = 0 ; i < NUM_OF_CLUSTERS ; i++)
	{
		//clusters[i].points = NULL;
		clusters[i].radius = 0;
		clusters[i].numOfPoints = 0;
	}
}

bool calculateClusterCenters(Cluster* cluster, const int NUM_OF_DIMENSIONS)
{
	printf("Num of points: %d\n", cluster->numOfPoints);
	bool returnValue = true;
	Point tempCenter;
	tempCenter.coordinates = (float*)calloc(NUM_OF_DIMENSIONS, sizeof(float));
	if(RUN_IN_PARALLEL)
	{
		omp_set_num_threads(cluster->numOfPoints);
 		#pragma omp parallel private(i)
 		#pragma omp parallel for
		for(int i = 0 ; i < cluster->numOfPoints ; i++)
		{
			for (int j = 0 ; j < NUM_OF_DIMENSIONS ; j++)
			{
				tempCenter.coordinates[j] += cluster->points[i].coordinates[j];
			}
		}
		calculateCenterUsingCuda(tempCenter.coordinates, NUM_OF_DIMENSIONS, cluster->numOfPoints);
	}
	else
	{
		for(int i = 0 ; i < cluster->numOfPoints ; i++)
		{
			for (int j = 0 ; j < NUM_OF_DIMENSIONS ; j++)
			{
				tempCenter.coordinates[j] += cluster->points[i].coordinates[j];
			}
		}
		for(int i = 0 ; i < NUM_OF_DIMENSIONS ; i++)
		{
			tempCenter.coordinates[i] = tempCenter.coordinates[i] / cluster->numOfPoints;
			for(int j = 0 ; j < NUM_OF_DIMENSIONS*3 ; j++)
			{
				exp(exp(-2.));
			}
		}
	}
	cluster->center = tempCenter;
	return returnValue;
}

double calculateClusterRadius(Cluster* cluster, const int NUM_OF_DIMENSIONS)
{
	//printf("Num of points: %d\n", cluster->numOfPoints);
	double tempRadius = 0;
	/*if(RUN_IN_PARALLEL)
	{
		float* temp = calculateClusterD(cluster, NUM_OF_DIMENSIONS);
		for(int i = 0 ; i < cluster->numOfPoints * cluster->numOfPoints ; i++)
		{
			printf("%f\n", temp[i]);
			if( temp[i] > tempRadius)
			{
				tempRadius =  temp[i];
			}
		}
		tempRadius = max(temp, cluster->numOfPoints, cluster->numOfPoints);
	}
	else
	{*/
		for(int i = 0 ; i < cluster->numOfPoints ; i++)
		{
			for(int j = 0 ; j < cluster->numOfPoints ; j++)
			{
				if(i != j)
				{
					double calculatedRadius = getDistanceBetweenTwoPoints(cluster->points[i], cluster->points[j], NUM_OF_DIMENSIONS);
					if(calculatedRadius > tempRadius)
					{
						tempRadius = calculatedRadius;
					}
				}
			}
		}
	//}
	
	return tempRadius;
}

void printPoint(Point* point, const int NUM_OF_DIMENSIONS)
{
	for(int i = 0 ; i < NUM_OF_DIMENSIONS ; i++)
	{
		printf("%f ", point->coordinates[i]);
	}
	printf("\n");
}

void checkIsCurrentClustersEnough(Point* points, Cluster* clusters, const int NUM_OF_DIMENSIONS, const int NUM_OF_PRODUCTS, const int NUM_OF_CLUSTERS, const int MAX_NUM_OF_ITERATION)
{
	for(int i = 0 ; i < MAX_NUM_OF_ITERATION ; i++)
	{
		printf("Amount of clusters: %d, Iteration number: %d\n", NUM_OF_CLUSTERS, i);
		addPointsToClusters(points, clusters, NUM_OF_PRODUCTS, NUM_OF_CLUSTERS, NUM_OF_DIMENSIONS);
		for(int j = 0 ; j < NUM_OF_CLUSTERS ; j++)
		{
			calculateClusterCenters(&clusters[j], NUM_OF_DIMENSIONS);
		}
		bool flag = isNeedToCalculateClusterCenter(clusters, NUM_OF_DIMENSIONS, NUM_OF_CLUSTERS);

		if(flag == false)
		{
			return;
		}
		else
		{
			removePointFromCluster(clusters, NUM_OF_CLUSTERS);
		}
	}
}

double calculateQM(Cluster* clusters, int NUM_OF_DIMENSIONS, int NUM_OF_CLUSTERS)
{
	double tempQM = 0;

	for(int i = 0 ; i < NUM_OF_CLUSTERS ; i++)
	{
		double d = (calculateClusterRadius(&clusters[i], NUM_OF_DIMENSIONS));
		for(int j = 0 ; j < NUM_OF_CLUSTERS ; j++)
		{
			if (i != j)
			{
				double dd = getDistanceBetweenTwoPoints(clusters[i].center, clusters[j].center, NUM_OF_DIMENSIONS);
				tempQM += d / dd;
			}
		}
	}
	return tempQM / (NUM_OF_CLUSTERS*(NUM_OF_CLUSTERS-1));
}

bool isNeedToCalculateClusterCenter(Cluster* clusters, const int NUM_OF_DIMENSIONS, const int NUM_OF_CLUSTERS)
{
	int clusterIndex = 0;

	for(int i = 0 ; i < NUM_OF_CLUSTERS ; i++)
	{
		for(int j = 0 ; j< clusters[i].numOfPoints ; j++)
		{
			for(int p = 0 ; p < NUM_OF_CLUSTERS ; p++)
			{
				if(p!=i)
				{
					float distanceToOtherClusterCenter = getDistanceBetweenTwoPoints(clusters[i].points[j], clusters[p].center, NUM_OF_DIMENSIONS);
					float distanceToPointClusterCenter = getDistanceBetweenTwoPoints(clusters[i].points[j], clusters[i].center, NUM_OF_DIMENSIONS);
					if (distanceToOtherClusterCenter < distanceToPointClusterCenter)
					{
						//printf("Other = %f, Now = %f\n", distanceToOtherClusterCenter, distanceToPointClusterCenter);
						return true;
					}
				}
			}
		}
	}
	return false;
}

void printOutputFile(Cluster* clusters, const int NUM_OF_CLUSTERS, const int NUM_OF_DIMENSIONS, const double QM)
{
	FILE* file = fopen(pathToFileOutput, "w");
	int rc = fprintf(file, "%d,%f\n", NUM_OF_CLUSTERS, QM);
	for(int i = 0 ; i < NUM_OF_CLUSTERS ; i++)
	{
		fprintf(file, "C%d,", (i+1));
		for(int j = 0 ; j < NUM_OF_DIMENSIONS ; j++)
		{
			if(j == NUM_OF_DIMENSIONS - 1)
			{
				fprintf(file, "%f\n", clusters[i].center.coordinates[j]);
			}
			else
			{
				fprintf(file, "	%f,", clusters[i].center.coordinates[j]);
			}
		}
	}
}

double max(float A[], int i, int j)
{
    int idx;
    double max_val = 0; /* = 0 not needed according to Jim Cownie comment */

    #pragma omp parallel for reduction(max:max_val) 
    for (idx = i; idx < j; idx++)
       max_val = max_val > A[idx] ? max_val : A[idx];

    return max_val;
}