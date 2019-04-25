#pragma once

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <vector>
#include <math.h>
#include <float.h>
#include <chrono>
#include <fstream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define AMOUNT 20
#define RANGE 50

#define TILE_DIM 32
#define roundup(numerator, denominator) (numerator+denominator-1/denominator)

class Point
{
public:
	__host__ __device__ Point()
	{
		x = NULL;
		y = NULL;
		distanceFromOrigin = NULL;
	}

	__host__ __device__ Point(float _x, float _y)
	{
		x = _x;
		y = _y;
		distanceFromOrigin = sqrt(x*x + y * y);
	}

	float x;
	float y;
	float distanceFromOrigin;

	__host__ __device__ bool Point::operator==(const Point & other)
	{
		if (other.x == this->x && other.y == this->y)
			return true;
		else
			return false;
	}

	// Point Inequality
	__host__ __device__ bool Point::operator!=(const Point & other)
	{
		return !(*this == other);
	}

	// To the right of other
	__host__ __device__ bool Point::operator>>(const Point& other)
	{
		if (other.x < this->x)
			return true;
		else
			return false;
	}

	// To the left of other
	__host__ __device__ bool Point::operator<<(const Point& other)
	{
		if (other.x > this->x)
			return true;
		else
			return false;
	}

	// Above other
	__host__ __device__ bool Point::operator>(const Point& other)
	{
		if (other.y > this->y)
			return true;
		else
			return false;
	}

	// Below other
	__host__ __device__ bool Point::operator<(const Point& other)
	{
		if (other.y < this->y)
			return true;
		else
			return false;
	}
};

class Edge
{
public:
	__host__ __device__ Edge()
	{

	}

	__host__ __device__ Edge(Point &_pointA, Point &_pointB)
	{
		pointA = _pointA;
		pointB = _pointB;

		midpoint.x = (pointB.x + pointA.x) / 2;
		midpoint.y = (pointB.y + pointA.y) / 2;

		rise = (pointB.y - pointA.y);
		run = (pointB.x - pointA.x);

		if (rise == 0)
			perpendicularSlope = FLT_MAX;
		if (run == 0)
			perpendicularSlope = 0;
		if (rise != 0 && run != 0)
			perpendicularSlope = -1 / (rise / run);

		yIntercept = midpoint.y - (midpoint.x*perpendicularSlope);

	}

	Point pointA;
	Point pointB;
	Point midpoint;

	double perpendicularSlope;
	double yIntercept;
	double rise;
	double run;

	// Returns the double length of the edge
	__host__ __device__ double Edge::getLength()
	{
		double v = abs(pointA.x - pointB.x);
		double u = abs(pointA.y - pointB.y);

		double length = sqrt(v*v + u * u);

		return length;
	}
};

class Triangle
{
public:
	Triangle()
	{
		edgeA = Edge(Point(), Point());
		edgeB = Edge(Point(), Point());
		edgeC = Edge(Point(), Point());
	}

	__host__ __device__ Triangle(Point &_pointA, Point &_pointB, Point &_pointC)
	{
		edgeA = Edge(_pointA, _pointB);
		edgeB = Edge(_pointB, _pointC);
		edgeC = Edge(_pointC, _pointA);

		//circumDistance = calculateCircumDistance(_pointA, _pointB, _pointC, circumCenter);

		if (circumDistance == 0)
			circumDistance = FLT_MAX;
	}

	Edge edgeA;
	Edge edgeB;
	Edge edgeC;

	Point circumCenter;

	double circumDistance;

	// Calculates the circumCenter of the triangle and returns the double circumDistance or Radius
	__host__ __device__ void Triangle::calculateCircumDistance()
	{
		double slopeA = edgeA.perpendicularSlope, slopeB = edgeB.perpendicularSlope;
		double yCoefficientA = 1, yCoefficientB = 1;
		double yInterceptA = edgeA.yIntercept, yInterceptB = edgeB.yIntercept;

		double determinant = slopeA * yCoefficientB - slopeB * yCoefficientA;

		if (determinant != 0)
		{

			circumCenter = Point((yCoefficientB * yInterceptA - yCoefficientA * yInterceptB) / -determinant, (slopeA * yInterceptB - slopeB * yInterceptA) / determinant);

			Edge circumEdge(circumCenter, edgeA.pointA);

			circumDistance = circumEdge.getLength();
		}
	}
};

int factorial(int n);

unsigned nChoosek(unsigned n, unsigned k);

std::vector<int> generateRandomNumbers(int max, int amount);

bool distinct(int number, std::vector<int> numbers, int currentIndex);

void combinations(std::vector<Triangle> &tVec, std::vector<Point> pArray);

void combinations(float pCombinationsX[], float pCombinationsY[], std::vector<Point> pArray);

void DelaunayTriangulation(std::vector<Point> pArray);

void GPUDelaunayTriangulation(std::vector<Point> pArray);

int calculateBlockSize(int blockWidth);
int calculateGridSize(int & block);

__global__ void triangulationKernel(float* d_pArrayX, float* d_pArrayY, float* d_pCombinationsX, float* d_pCombinationsY, Triangle* d_tArray, int arraySize, int blockWidth);

int main(int argc, char *argv[])
{
	std::vector<int> x = generateRandomNumbers(AMOUNT, RANGE);
	_sleep(3000);
	std::vector<int> y = generateRandomNumbers(AMOUNT, RANGE);

	std::vector<Point> pArray;

	int size = (int)x.size() - 1;
	for (int i = 0; i < (int)x.size(); i++)
	{
		pArray.push_back(Point(x[i], y[size--]));
	}

	//std::vector<Point> pArray = { Point(1,2), Point(3,5), Point(2,9), Point(8,3), Point(6,9), Point(10,4), Point(13,8), Point(11,10), Point(4,8), Point(6,1) };

	//auto startClock = std::chrono::high_resolution_clock::now();
	//DelaunayTriangulation(pArray);
	//auto stopClock = std::chrono::high_resolution_clock::now();

	//auto time = std::chrono::duration_cast<std::chrono::microseconds>(stopClock - startClock);

	//std::cout << "execution time: " << time.count() << "us" << std::endl << std::endl;

	GPUDelaunayTriangulation(pArray);

	return 0;
}

int factorial(int n)
{
	if (n > 1)
		return n * factorial(n - 1);
	else
		return 1;
}

unsigned nChoosek(unsigned n, unsigned k)
{
	if (k > n) return 0;
	if (k * 2 > n) k = n - k;
	if (k == 0) return 1;

	int result = n;
	for (int i = 2; i <= k; ++i) {
		result *= (n - i + 1);
		result /= i;
	}
	return result;
}

std::vector<int> generateRandomNumbers(int amount, int max)
{
	srand(time(0));

	std::vector<int> numbers;
	int number;

	for (int i = 0; i < amount; i++)
	{
		numbers.push_back((rand() % max) + 1);
		while (!distinct(numbers[i], numbers, i))
		{
			if (numbers[i] < max)
				numbers[i]++;
			else
				numbers[i]--;
		}
	}

	return numbers;
}

bool distinct(int number, std::vector<int> numbers, int currentIndex)
{
	for (int i = 0; i < (int)numbers.size(); i++)
	{
		if (number == numbers[i] && i != currentIndex)
		{
			return false;
		}
	}
}

void combinations(std::vector<Triangle> &tVec, std::vector<Point> pArray)
{
	int size = (int)pArray.size();
	int count = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			for (int k = j + 1; k < size; k++)
			{

				Triangle newTriangle(pArray[i], pArray[j], pArray[k]);

				newTriangle.calculateCircumDistance();

				bool add = true;
				for (int l = 0; l < size; l++)
				{
					if (newTriangle.edgeA.pointA != pArray[l] && newTriangle.edgeB.pointA != pArray[l] && newTriangle.edgeC.pointA != pArray[l])
					{
						float v = abs(newTriangle.circumCenter.y - pArray[l].y);
						float u = abs(newTriangle.circumCenter.x - pArray[l].x);
						float distance = sqrt((v*v) + (u*u));

						if (distance < newTriangle.circumDistance)
						{
							add = false;
							break;
						}
					}
				}

				if (add)
					tVec.push_back(newTriangle);

				count++;
			}
		}
	}
	std::cout << count << std::endl;
}

void combinations(float pCombinationsX[], float pCombinationsY[], std::vector<Point> pArray)
{
	int size = (int)pArray.size();
	int l = 0, m = 0;

	int count = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			for (int k = j + 1; k < size; k++)
			{
				pCombinationsX[l++] = pArray[i].x;
				pCombinationsX[l++] = pArray[j].x;
				pCombinationsX[l++] = pArray[k].x;

				pCombinationsY[m++] = pArray[i].y;
				pCombinationsY[m++] = pArray[j].y;
				pCombinationsY[m++] = pArray[k].y;

				count++;
			}
		}
	}

	std::cout << count << std::endl;
}

void DelaunayTriangulation(std::vector<Point> pArray)
{
	std::cout << "\n\n####################### CPU Delaunay Triangulation #######################" << std::endl << std::endl;

	int size = (int)pArray.size();

	std::vector<Triangle> tVec;

	combinations(tVec, pArray);

	std::cout << std::endl;

	for (int i = 0; i < (int)tVec.size(); i++)
	{
		std::cout << " " << i << "\t(" << tVec[i].edgeA.pointA.x << ", " << tVec[i].edgeA.pointA.y << "), "
			<< "(" << tVec[i].edgeB.pointA.x << ", " << tVec[i].edgeB.pointA.y << "), "
			<< "(" << tVec[i].edgeC.pointA.x << ", " << tVec[i].edgeC.pointA.y << ")" << std::endl << std::endl;
	}
}

void GPUDelaunayTriangulation(std::vector<Point> pArray)
{
	std::cout << "\n\n####################### GPU Delaunay Triangulation #######################" << std::endl << std::endl;

	// Handle the input vector of initial points by converting to two floating point coordinate arrays on the host
	int size = (int)pArray.size();
	int inputArraySize = (size * sizeof(float));
	float *h_pArrayX = new float[size];
	float *h_pArrayY = new float[size];
	for (int i = 0; i < size; i++)
	{
		h_pArrayX[i] = pArray[i].x;
		h_pArrayY[i] = pArray[i].y;
	}

	// Prepare the space on the device and copy them over
	float* d_pArrayX, *d_pArrayY;
	cudaMalloc((void**)&d_pArrayX, inputArraySize);
	cudaMalloc((void**)&d_pArrayY, inputArraySize);
	cudaMemcpy(d_pArrayX, h_pArrayX, inputArraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pArrayY, h_pArrayY, inputArraySize, cudaMemcpyHostToDevice);

	// Generate all possible combinations of the points to be handled by the GPU
	// store them in two integer coordinate arrays on the host
	int blockWidth = nChoosek((int)pArray.size(), 3);
	int inputComboSize = blockWidth * 3 * sizeof(float);
	float *h_pCombinationsX = new float[blockWidth * 3], *h_pCombinationsY = new float[blockWidth * 3];
	combinations(h_pCombinationsX, h_pCombinationsY, pArray);

	// Prepare the space on the device and copy them over
	float* d_pCombinationsX, *d_pCombinationsY;
	cudaMalloc((void**)&d_pCombinationsX, inputComboSize);
	cudaMalloc((void**)&d_pCombinationsY, inputComboSize);
	cudaMemcpy(d_pCombinationsX, h_pCombinationsX, inputComboSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pCombinationsY, h_pCombinationsY, inputComboSize, cudaMemcpyHostToDevice);

	// Prepare the space to store the triangles on the device and on the host
	Triangle* d_tArray;
	cudaMalloc((void**)&d_tArray, blockWidth * sizeof(Triangle));
	cudaMemcpy(d_tArray, 0, blockWidth*sizeof(Triangle), cudaMemcpyHostToDevice);

	// Calculate the dimensions of block and grid
	int block = calculateBlockSize(blockWidth);
	int grid = calculateGridSize(block);

	if (block > 0)
		int thing;

	triangulationKernel<<<grid, block>>>(d_pArrayX, d_pArrayY, d_pCombinationsX, d_pCombinationsY, d_tArray, size, blockWidth);


	Triangle* h_tArray = new Triangle[blockWidth];

	cudaMemcpy(h_tArray, d_tArray, blockWidth*sizeof(Triangle), cudaMemcpyDeviceToHost);

	std::ofstream outputPoints("points.txt");
	std::ofstream outputTriangles("triangles.txt");

	for (int i = 0; i < size; i++)
	{
		outputPoints << pArray[i].x << " " << pArray[i].y << std::endl;
	}

	int count = 0, outputCount = 0;
	for (int i = 0; i < blockWidth; i++)
	{
		if (h_tArray[i].edgeA.pointA.x != 0)
		{
			
			for (int j = 0; j < size; j++)
			{
				if (h_tArray[i].edgeA.pointA == pArray[j])
				{
					if (outputCount != 2)
					{
						outputTriangles << j + 1 << " ";
						outputCount++;
					}
					else
					{
						outputCount = 0;
						outputTriangles << j  +1 << ";" << std::endl;
					}
				}

				if (h_tArray[i].edgeB.pointA == pArray[j])
				{
					if (outputCount != 2)
					{
						outputTriangles << j + 1 << " ";
						outputCount++;
					}
					else
					{
						outputCount = 0;
						outputTriangles << j + 1 << ";" << std::endl;
					}
				}

				if (h_tArray[i].edgeC.pointA == pArray[j])
				{
					if (outputCount != 2)
					{
						outputTriangles << j + 1 << " ";
						outputCount++;
					}
					else
					{
						outputCount = 0;
						outputTriangles << j + 1 << ";" << std::endl;
					}
				}
			}
			

			std::cout << " " << count++ << "\t(" << h_tArray[i].edgeA.pointA.x << ", " << h_tArray[i].edgeA.pointA.y << "), "
				<< "(" << h_tArray[i].edgeB.pointA.x << ", " << h_tArray[i].edgeB.pointA.y << "), "
				<< "(" << h_tArray[i].edgeC.pointA.x << ", " << h_tArray[i].edgeC.pointA.y << ")" << std::endl << std::endl;

		}
	}

	outputPoints.close();
	outputTriangles.close();

	// Free all allocated memory
	delete[] h_pArrayX, h_pArrayY, h_pCombinationsX, h_pCombinationsY, h_tArray;
	cudaFree(d_pArrayX); cudaFree(d_pArrayY); cudaFree(h_pCombinationsX); cudaFree(h_pCombinationsY); cudaFree(d_tArray);
}

int calculateBlockSize(int blockWidth)
{
	return roundup(blockWidth, TILE_DIM);
}

int calculateGridSize(int & block)
{
	int grid = (block + 1023) / 1024;
	block = (block + grid - 1) / grid;

	return grid;
}

__global__ void triangulationKernel(float* d_pArrayX, float* d_pArrayY, float* d_pCombinationsX, float* d_pCombinationsY, Triangle* d_tArray, int arraySize, int blockWidth)
{

	if (threadIdx.x > blockWidth)
		return;

	int globalThreadID = blockDim.x*blockIdx.x + threadIdx.x;

	int index = globalThreadID * 3;

	Triangle newTriangle(Point(d_pCombinationsX[index], d_pCombinationsY[index]), Point(d_pCombinationsX[index + 1], d_pCombinationsY[index + 1]), Point(d_pCombinationsX[index + 2], d_pCombinationsY[index + 2]));

	newTriangle.calculateCircumDistance();

	bool add = true;
	for (int i = 0; i < arraySize; i++)
	{
		Point currentPoint(d_pArrayX[i], d_pArrayY[i]);
		if (newTriangle.edgeA.pointA != currentPoint && newTriangle.edgeB.pointA != currentPoint && newTriangle.edgeC.pointA != currentPoint)
		{
			float v = abs(newTriangle.circumCenter.y - currentPoint.y);
			float u = abs(newTriangle.circumCenter.x - currentPoint.x);
			float distance = sqrt((v*v) + (u*u));

			if (distance < newTriangle.circumDistance)
			{
				add = false;
				break;
			}
		}
	}

	if (add)
	{
		d_tArray[threadIdx.x] = newTriangle;
	}
}