#pragma once
#include "Geo.hpp"
#include <math.h>
#include <algorithm>

// Point Equality
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

// Returns the double length of the edge
__host__ __device__ double Edge::getLength()
{
	double v = abs(pointA.x - pointB.x);
	double u = abs(pointA.y - pointB.y);

	double length = sqrt(v*v + u*u);

	return length;
}

// Calculates the circumCenter of the triangle and returns the double circumDistance or Radius
__host__ __device__ void Triangle::calculateCircumDistance()
{
	//Edge edgePQ(A, B), edgePR(A, C);

	double a1 = edgeA.perpendicularSlope, a2 = edgeB.perpendicularSlope;
	double b1 = 1, b2 = 1;
	double c1 = edgeA.yIntercept, c2 = edgeB.yIntercept;

	double determinant = a1 * b2 - a2 * b1;

	if (determinant != 0)
	{

		circumCenter = Point((b2 * c1 - b1 * c2) / -determinant, (a1 * c2 - a2 * c1) / determinant);

		Edge circumEdge(circumCenter, edgeA.pointA);

		circumDistance = circumEdge.getLength();
	}
}