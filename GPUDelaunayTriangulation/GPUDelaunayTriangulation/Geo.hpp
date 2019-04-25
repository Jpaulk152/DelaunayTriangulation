#pragma once

#include <math.h>
#include <float.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

	__host__ __device__ bool Point::operator==(const Point& other); // Point Equality
	__host__ __device__ bool Point::operator!=(const Point& other);	// Point Inequality
	__host__ __device__ bool Point::operator>>(const Point& other); // To the right of other
	__host__ __device__ bool Point::operator<<(const Point& other);	// To the left of other
	__host__ __device__ bool Point::operator>(const Point& other);	// Above other
	__host__ __device__ bool Point::operator<(const Point& other);	// Below other
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

	double getLength();
};

class Triangle
{
public:
	Triangle()
	{

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

	__host__ __device__ void calculateCircumDistance();
};
