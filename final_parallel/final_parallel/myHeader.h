#ifndef MYHEADER_H
#define MYHEADER_H
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

int NUM_OF_CLUSTERS = 2;
Point *readDataFromFile(int* NUM_OF_DIMENSIONS, int* NUM_OF_PRODUCTS, int* MAX_NUM_OF_CLUSTERS, int* MAX_NUM_OF_ITERATION, float* QM, Cluster* clusters, const int NUM_OF_CLUSTERS);
void addPointsToClusters(Point* points, Cluster* clusters, const int NUM_OF_PRODUCTS, const int NUM_OF_CLUSTERS, const int NUM_OF_DIMENSIONS);
double getDistance(Point point, Cluster cluster, const int NUM_OF_DIMENSIONS);
bool calculateClusterCenters(Cluster* cluster, const int NUM_OF_DIMENSIONS);
double calculateClusterRadius(Cluster* cluster, const int NUM_OF_DIMENSIONS);
void removePointFromCluster(Cluster* clusters, const int NUM_OF_CLUSTERS);
bool isNeedToCalculateClusterCenter(Cluster* clusters, const int NUM_OF_DIMENSIONS, const int NUM_OF_CLUSTERS);
double getDistanceBetweenTwoPoints(Point point1, Point point2, const int NUM_OF_DIMENSIONS);
void printPoint(Point* point, const int NUM_OF_DIMENSIONS);
void checkIsCurrentClustersEnough(Point* points, Cluster* clusters, const int NUM_OF_DIMENSIONS, const int NUM_OF_PRODUCTS, const int NUM_OF_CLUSTERS, const int MAX_NUM_OF_ITERATION);
double calculateQM(Cluster* clusters, int NUM_OF_DIMENSIONS, int NUM_OF_CLUSTERS);
Cluster* appendPointsAsClusters(Cluster* clusters, Point* points, int* NUM_OF_DIMENSIONS, const int NUM_OF_CLUSTERS);
float* calculateCenterUsingCuda(float* coordinates, const int NUM_OF_DIMENSIONS, const int NUM_OF_POINTS);
//const char* pathToFile = "C:/Shay Zambrovski/kmean/Sales_Transactions_Dataset_Weekly.dat";
const char* pathToFile = "E:/College/PComputing/Project-Parallel/Sales_Transactions_Dataset_Weekly.dat";
#endif