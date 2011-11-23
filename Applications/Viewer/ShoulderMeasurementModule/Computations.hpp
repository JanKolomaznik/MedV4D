#ifndef SHOULDER_MEASUREMENT_COMPUTATIONS_HPP
#define SHOULDER_MEASUREMENT_COMPUTATIONS_HPP


#include "MedV4D/Common/Common.h"
#include "GUI/utils/QtModelViewTools.h"

typedef VectorItemModel< M4D::Point3Df > PointSet;

struct HeadMeasurementData
{
	HeadMeasurementData(): available( false )
	{}

	bool available;

	Vector3f point;
	Vector3f normal;
	Vector3f vDirection;
	Vector3f wDirection;
};


void
getHeadMeasurementData( const PointSet &aPoints, HeadMeasurementData &aHeadMeasurementData );

struct ProximalShaftMeasurementData
{
	ProximalShaftMeasurementData(): available( false )
	{}

	bool available;

	Vector3f point;
	Vector3f centerPoint;
	Vector3f direction;
	float height;
	float radius;


	Vector3f bboxP1;
	Vector3f bboxP2;

	Vector3f minimum;
	Vector3f maximum;
};

void
getProximalShaftMeasurementData( const PointSet &aPoints, ProximalShaftMeasurementData &aProximalShaftMeasurementData );


#endif /*SHOULDER_MEASUREMENT_COMPUTATIONS_HPP*/
