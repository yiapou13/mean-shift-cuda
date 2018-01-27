//VERSION 1.0 stable


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <device_functions.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <Windows.h>

#define bandwidth 1
#define iterations 7
#define density 1
#define filename "test_big.txt"
#define rows 4800
#define columns 2
#define THREADSperBLOCK 1024

__device__ void euclidean_distance_1d(double *points, double *temp_points, int point_a, int point_b, double *distance)
{
	double total = 0;

	for (int i = 0; i < columns; i++) // itterations: number of coordinates -> here 2 (x,y)
	{
	double temp = pow((points[point_a * columns + i] - temp_points[point_b * columns + i]), 2); //index for 1d table with formation: [X1, Y1, X2, Y2,..., Xn, Yn]
	total += temp;
	}

	*distance = total;
}

__device__ void gaussian_kernel(double *distance, double *weight)
{
	*weight = exp(-1.0 / 2.0 * (*distance) * (*distance) / (bandwidth*bandwidth));
}

__global__ void meanshift(double *d_points, double *d_testpoints, double *final_points)
{
	__shared__ double neighbours[THREADSperBLOCK * columns];
	__shared__ double t_final_points[THREADSperBLOCK * columns];
	__shared__ double t_temp_points[THREADSperBLOCK * columns];

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int t_index = threadIdx.x;
	
	if (t_index < THREADSperBLOCK  && index < rows * columns)
	{
		neighbours[t_index * columns] = d_points[threadIdx.x * columns + THREADSperBLOCK * columns * blockIdx.x];
		neighbours[t_index * columns + 1] = d_points[threadIdx.x * columns + THREADSperBLOCK * columns * blockIdx.x + 1];

		t_temp_points[t_index * columns] = d_testpoints[threadIdx.x * columns + THREADSperBLOCK * columns * blockIdx.x];
		t_temp_points[t_index * columns + 1] = d_testpoints[threadIdx.x * columns + THREADSperBLOCK * columns * blockIdx.x + 1];
	}

	__syncthreads();
	if (t_index == 0)printf("%d\n", gridDim.x); // __syncthreads() seems to not working properly, synchronizing with printf
	
	if (index < rows * columns)
	{	
		double distance = 0, weight = 0, total_weight = 0;
		int i;


		for (i = 0; i < THREADSperBLOCK; i++)
		{
			euclidean_distance_1d(t_temp_points, neighbours, t_index, i, &distance);
			if (distance <= density && distance != 0 )
			{
				gaussian_kernel(&distance, &weight);
				t_final_points[t_index * columns] += weight * neighbours[i * columns];
				t_final_points[t_index * columns + 1] += weight * neighbours[i * columns + 1];
				total_weight += weight;
			}
		}

		t_final_points[t_index * columns] = t_final_points[t_index * columns] / total_weight;
		t_final_points[t_index * columns + 1] = t_final_points[t_index * columns + 1] / total_weight;
	}

	memcpy(&final_points[blockIdx.x * THREADSperBLOCK * columns], t_final_points, THREADSperBLOCK * columns * sizeof(double));

	return;

}

int main()
{
	//Declarations
	double *meanshift_handler(double *, double *, float *);
	double *check(double *);
	double euclidean_distance(double *, double *, int, int);

	FILE *fp;
	char line[16];
	double value;
	double *testpoints = (double*)malloc(rows * columns * sizeof(double));
	double *points = (double*)malloc(rows * columns * sizeof(double));
	int *clusters = (int*)calloc(rows, sizeof(int)); //initialize array with zeros
	int i = 0;
	float miliseconds = 0;
	float total = 0;

	//Reading points from file
	fp = fopen(filename, "rt");
	while (fgets(line, 16, fp) != NULL)
	{
		sscanf(line, "%lf", &value);

		testpoints[i] = value;
		points[i] = value;
		i++;

	}
	fclose(fp);

	//Mean shift process, 
	for (i = 0; i < iterations; i++)
	{
		testpoints = meanshift_handler(points, testpoints, &miliseconds);
		cudaDeviceSynchronize();
		total += miliseconds;
	}
	
	//Clustering process
	int cluster = 1;

	for (i = 0; i < rows; i++)
	{
		if (clusters[i] == 0) //check if a point belongs already to a cluster
		{
			clusters[i] = cluster;
			for (int j = 0; j < rows; j++)
			{
				if (euclidean_distance(testpoints, testpoints, i, j) <= 1) // cluster points with distance less than 1
				{
					clusters[j] = cluster;
				}
			}

			cluster++;
		}
	}

	//Check clustering results
	int matches = 0;
	fp = fopen("labels.txt", "rt");
	i = 0;

	while (fgets(line, 16, fp) != NULL)
	{
		sscanf(line, "%lf", &value);

		if (clusters[i] == value) matches++;

		i++;
	}

	printf("Matches: %d", matches);
	fclose(fp);

	//Writing final points
	fp = fopen("results1.txt", "w");
	for (i = 0; i < rows; i++)
		fprintf(fp,"%f\t%f\n", testpoints[i * columns], testpoints[i * columns + 1]);
	fprintf(fp, "Time: %f\n", total);
	fclose(fp);

	//Writing labels(clusters) of each point
	fp = fopen("labels.txt", "w");
	for (i = 0; i < rows; i++)
		fprintf(fp, "%d\n", clusters[i]);
	fclose(fp);

	free(testpoints);

	return 0;
}

double *meanshift_handler(double *points, double *testpoints, float *miliseconds)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double *d_points, *d_testpoints, *final_points; // CUDA variables
	int gridsize = rows * columns / THREADSperBLOCK;

	cudaMalloc((void**)&d_points, rows * columns * sizeof(double));

	cudaMalloc((void**)&d_testpoints, rows * columns * sizeof(double));

	cudaMalloc((void**)&final_points, rows * columns * sizeof(double));

	cudaMemcpy(d_points, points, rows * columns * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_testpoints, testpoints, rows * columns * sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaEventRecord(start);
	meanshift << <ceil(gridsize) + 1, THREADSperBLOCK >> >  (d_points, d_testpoints, final_points);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();

	cudaMemcpy(testpoints, final_points, rows * columns * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(miliseconds, start, stop);

	cudaFree(d_points);
	cudaFree(d_testpoints);
	cudaFree(final_points);

	return testpoints;
}

double euclidean_distance(double *points, double *temp_points, int point_a, int point_b) // same as euclidean_distance_1d but in host
{
	double total = 0;

	for (int i = 0; i < columns; i++)
	{
		double temp = pow((points[point_a * columns + i] - temp_points[point_b * columns + i]), 2);
		total += temp;
	}

	return total;
}