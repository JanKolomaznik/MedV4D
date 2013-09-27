#ifndef MEASUREMENT_COMPUTATIONS_HPP
#define MEASUREMENT_COMPUTATIONS_HPP


#include "MedV4D/Common/Common.h"
#include "MedV4D/GUI/utils/QtModelViewTools.h"

typedef VectorItemModel< M4D::Point3Df > PointSet;

struct HeadMeasurementData
{
	HeadMeasurementData(): available( false )
	{}

	bool available;			///< Should be set if the data structure is filled with meaningful data.

	Vector3f point;			///< The center of the point-set.
	Vector3f normal;		///< Normal to the plane defined by the center-point and the 2 most varying directions of the point-set.
	Vector3f vDirection;	///< The most varying direction of the point-set.
	Vector3f wDirection;	///< The second most varying direction of the point-set.
};


/** \brief Compute the values of the HeadMeasurementData structure.
 *
 * \param aPoints	Point-set to use for the computation.
 * \param aHeadMeasurementData	Data structure to hold the results.
 *
 */
void
getHeadMeasurementData( const PointSet &aPoints, HeadMeasurementData &aHeadMeasurementData );

struct ProximalShaftMeasurementData
{
	ProximalShaftMeasurementData(): available( false )
	{}

	bool available;			///< Should be set if the data structure is filled with meaningful data.

	Vector3f point;			///< Base-point of the cylinder (lying on the "axis" of the point-set, sticking out by 10% from the minimal AABB to the lower end).
	Vector3f centerPoint;	///< Center-point of the cylinder (and the "axis" of the point set).
	Vector3f direction;		///< Direction vector of the cylinder (and the "axis" of the point-set, pointing to their "upper" end).
	float height;			///< Height of the cylinder (the length of the "axis" elongated by 20% -- 10% on the upper and the lower end respectively).
	float radius;			///< Radius of the cylinder (the distance of the nearest point of the point-set to the "axis").


	Vector3f bboxP1;		///< Lower end of the "axis" (where it intersects the minimal AABB to the point-set nearest to the origin).
	Vector3f bboxP2;		///< Upper end of the "axis" (where it intersects the minimal AABB to the point-set farthest from the origin).

	Vector3f minimum;		///< Minimum bounds to the point-set (the vertex of the minimal AABB to the set nearest to the origin).
	Vector3f maximum;		///< Maximum bounds to the point-set (the vertex of the minimal AABB to the set farthest from the origin).
};

/** \brief Compute the values of the ProximalShaftMeasurementData structure.
 *
 * \param aPoints	Point-set to use for the computation.
 * \param aProximalShaftMeasurementData	Data structure to hold the results.
 *
 */
void
getProximalShaftMeasurementData( const PointSet &aPoints, ProximalShaftMeasurementData &aProximalShaftMeasurementData );


#endif /*MEASUREMENT_COMPUTATIONS_HPP*/
