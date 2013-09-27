#include "MeasurementModule/Computations.hpp"
#include "MedV4D/Common/MathTools.h"
#include "MedV4D/Common/GeometricAlgorithms.h"

#include <numeric>
#include <cstdlib>

#include <eigen2/Eigen/Dense>


/** \brief Generate a random vector (bounded per component).
 *
 * \param aMin Lower bounds.
 * \param aMax Upper bounds.
 * \return The generated random vector.
 *
 */
template< typename TNumType, size_t tDim >
Vector< TNumType, tDim >
getRandomVector( const Vector< TNumType, tDim > &aMin, const Vector< TNumType, tDim > &aMax )
{
	Vector< TNumType, tDim > res;
	for ( size_t i = 0; i < tDim; ++i ) {
		res[i] = aMin[i] + ((static_cast<double>( rand() ) / RAND_MAX ) * (aMax[i] - aMin[i]));
	}
	return res;
}

template< typename TNumType, size_t tDim, typename TFunction >
class NumericOptimizer
{
public:
	NumericOptimizer(): mStepTestLimit( 20 ), mGlobalCycleLimit( 500 )
	{}

	/*Vector< TNumType, tDim >
	optimize( const Vector< TNumType, tDim > &aMin, const Vector< TNumType, tDim > &aMax, Vector< TNumType, tDim > aInitialValue, TFunction aFunction )
	{
		srand ( time(NULL) );

		Vector< TNumType, tDim > best = aInitialValue;
		TNumType bestScore = aFunction( aInitialValue );
		for( size_t i = 0; i < 5000; ++i ) {
			Vector< TNumType, tDim > tmpVec = getRandomVector< TNumType, tDim >( aMin, aMax );
			TNumType tmpVal = aFunction( tmpVec );
			if( tmpVal < bestScore ) {
				bestScore = tmpVal;
				best = tmpVec;
				LOG( "iteration " << i << "; score = " << bestScore );
			}
		}
		return best;
	}*/


    /** \brief Find a vector of input values which minimizes a given function between the given bounds.
     *
     * \param aMin Lower bounds for the result vector.
     * \param aMax Upper bounds for the result vector.
     * \param aInitialValue Start the search from this initial value.
     * \param aFunction Function to minimize.
     * \return The vector giving the lowest function value which has been found.
     *
     */
	Vector< TNumType, tDim >
	optimize( const Vector< TNumType, tDim > &aMin, const Vector< TNumType, tDim > &aMax, Vector< TNumType, tDim > aInitialValue, TFunction aFunction )
	{
		srand ( time(NULL) );
		Vector< TNumType, tDim > best = aInitialValue;
		TNumType bestScore = aFunction( aInitialValue );
		for( size_t k = 0; k < mGlobalCycleLimit; ++k ) {
			for( size_t i = 0; i < tDim; ++i ) {

				Vector< TNumType, tDim > tmpVec1 = best;
				Vector< TNumType, tDim > tmpVec2 = best;
				TNumType step = (static_cast<double>( rand() ) / RAND_MAX ) * (aMax[i] - aMin[i])/10;
				tmpVec1[i] += step;
				tmpVec2[i] -= step;
				TNumType s1 = aFunction( tmpVec1 );
				TNumType s2 = aFunction( tmpVec2 );
				size_t j = 0;
				while (s1 > bestScore && s2 > bestScore && j++ < mStepTestLimit) {
					step *= 0.5f;
					tmpVec1 = best;
					tmpVec2 = best;
					s1 = aFunction( tmpVec1 );
					s2 = aFunction( tmpVec2 );
				}
				if ( s1 < s2 && s1 < bestScore ) {
					bestScore = s1;
					best = tmpVec1;
				} else if ( s2 < bestScore ) {
					bestScore = s2;
					best = tmpVec2;
				}
			}
		}

		/*srand ( time(NULL) );

		Vector< TNumType, tDim > best = aInitialValue;
		TNumType bestScore = aFunction( aInitialValue );
		for( size_t i = 0; i < 5000; ++i ) {
			Vector< TNumType, tDim > tmpVec = getRandomVector< TNumType, tDim >( aMin, aMax );
			TNumType tmpVal = aFunction( tmpVec );
			if( tmpVal < bestScore ) {
				bestScore = tmpVal;
				best = tmpVec;
				LOG( "iteration " << i << "; score = " << bestScore );
			}
		}*/
		return best;
	}

	size_t mStepTestLimit;
	size_t mGlobalCycleLimit;
};


struct CylinderMaximumFillFtor
{
	CylinderMaximumFillFtor( const PointSet &aPoints ): mPoints( aPoints )
	{ }

    /** \brief Functor operator to find the minimum distance of a line to a set of points in 3D.
     *
     * \param aCoord Defines the line to consider (the first 3 values define a point on the line, the last 3 define a vector giving the direction of the line).
     * \return The distance of the nearest point to the line.
     *
     */
	float
	operator()( const Vector< float, 6 > &aCoord )
	{
		Vector3f center( aCoord[0], aCoord[1], aCoord[2] );
		Vector3f v1( aCoord[3], aCoord[4], aCoord[5] );
		VectorNormalization( v1 );

		float minDistance = M4D::PointLineDistance( static_cast< Vector3f >( mPoints[0] ), center, v1 );
		float sumDistance;
		for ( size_t i = 1; i < mPoints.size(); ++i ) {
			float tmp = M4D::PointLineDistance( static_cast< Vector3f >( mPoints[i] ), center, v1 );
			minDistance = M4D::min( minDistance, tmp );
			sumDistance += tmp;
		}

		//return sumDistance - mPoints.size()*minDistance;
		return -minDistance;
	}

	const PointSet &mPoints;
};


void
computeCovarianceMatrixFromPointSet( const PointSet &aPoints, Vector3f &aCenter, Eigen::Matrix3f &aCovarianceMatrix )
{
	float scale = 1.0f/aPoints.size();
	aCenter = scale * std::accumulate( aPoints.begin(), aPoints.end(), Vector3f() );

/*	//compute variances
	Vector3f variances;
	for ( size_t k = 0; k < aPoints.size(); ++k ) {
		variances += VectorCoordinateProduct( aPoints[k]-aCenter, aPoints[k]-aCenter );
	}
	variances *= scale;*/

	for ( size_t i = 0; i < 3; ++i ) {
		for ( size_t j = 0; j < 3; ++j ) {
			float covariance = 0.0f;
			for ( size_t k = 0; k < aPoints.size(); ++k ) {
				covariance += (aPoints[k][i]-aCenter[i]) * (aPoints[k][j]-aCenter[j]);
			}
			aCovarianceMatrix(i,j) = scale * covariance;
		}
	}
}

/** \brief Finds the nearest plane to a set of points, its normal, and the center of the point-set.
 *
 * The resulting plane is produced by finding the center of the set, and the 2 vectors giving the 2 most varying directions of the set (linearly independent eigenvectors).
 *
 * \param aPoints Point set to consider.
 * \param aHeadMeasurementData Structure to fill the results in (".point" is the center of the point-set, ".normal" is the normalized normal of the plane and ".vDirection" and ".wDirection" are the direction vectors).
 *
 */
void
getHeadMeasurementData( const PointSet &aPoints, HeadMeasurementData &aHeadMeasurementData )
{
	ASSERT( aPoints.size() >= 3 );

	Vector3f center;
	Eigen::Matrix3f covarianceMatrix;

	computeCovarianceMatrixFromPointSet( aPoints, center, covarianceMatrix );

	typedef Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> Solver;
	Solver eigensolver(covarianceMatrix);

	Solver::RealVectorType eigenVals = eigensolver.eigenvalues();
	Eigen::Matrix3f eigenVectors = eigensolver.eigenvectors();
	D_PRINT( "Eigen values :\n" << eigenVals );
	D_PRINT( "Eigen vectors :\n" << eigenVectors );

	// Get the 2 most varying directions.
	Vector3f v1( eigenVectors(0,2), eigenVectors(1,2), eigenVectors(2,2) );
	Vector3f v2( eigenVectors(0,1), eigenVectors(1,1), eigenVectors(2,1) );

/*	Vector3f v1 = mHumeralHeadPoints[0] - center;
	Vector3f v2 = mHumeralHeadPoints[1] - center;
	VectorNormalization( v1 );
	VectorNormalization( v2 );*/
	Vector3f normal = VectorProduct( v1, v2 );
	VectorNormalization( normal );
	/*v2 = VectorProduct( v1, normal );
	VectorNormalization( v2 );*/

	aHeadMeasurementData.point = center;	// The center of the point-set.
	aHeadMeasurementData.normal = normal;	// Normal to the plane defined by the center-point and the 2 most varying directions of the point-set.
	aHeadMeasurementData.vDirection = v1;	// The most varying direction of the point-set.
	aHeadMeasurementData.wDirection = v2;	// The second most varying direction of the point-set.
	aHeadMeasurementData.available = true;	// The data structure is filled with meaningful data.
}


/** \brief
 *
 * \param aPoints Point set to consider.
 * \param aProximalShaftMeasurementData Structure to fill the results in ().
 *
 */
void
getProximalShaftMeasurementData( const PointSet &aPoints, ProximalShaftMeasurementData &aProximalShaftMeasurementData )
{
	Vector3f center;
	Vector3f v1;
	Eigen::Matrix3f covarianceMatrix;


	// Get the minimum and maximum bounds of the point set.
	Vector3f minimum = aPoints[0];
	Vector3f maximum = aPoints[0];
	for ( size_t i = 1; i < aPoints.size(); ++i ) {
		minimum = M4D::minVect< float, 3 >( static_cast< const Vector3f &>( minimum ), static_cast< const Vector3f &>( aPoints[i] ) );
		maximum = M4D::maxVect< float, 3 >( static_cast< const Vector3f &>( maximum ), static_cast< const Vector3f &>( aPoints[i] ) );
		/*D_PRINT( "point : " << aPoints[i] );
		D_PRINT( "min : " << minimum );
		D_PRINT( "max : " << maximum << "\n" );*/
	}

	// Find the center of the point-set and its most varying direction, or use the values from a previous computation if available.
	if ( !aProximalShaftMeasurementData.available ) {
		computeCovarianceMatrixFromPointSet( aPoints, center, covarianceMatrix );
		typedef Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> Solver;
		Solver eigensolver(covarianceMatrix);

		Solver::RealVectorType eigenVals = eigensolver.eigenvalues();
		Eigen::Matrix3f eigenVectors = eigensolver.eigenvectors();
		D_PRINT( "Eigen values :\n" << eigenVals );
		D_PRINT( "Eigen vectors :\n" << eigenVectors );

		v1 = Vector3f( eigenVectors(0,2), eigenVectors(1,2), eigenVectors(2,2) );

	} else {
		center = aProximalShaftMeasurementData.centerPoint;
		v1 = aProximalShaftMeasurementData.direction;
	}
	//********************************************************************
	// Try to find (fine-tune) the "axis" of the point-set by searching for the line nearest to all points in the set,
	// starting from the line given by the center-point and the direction of the most significant variance.
	// Do this by testing the line's distance to the set after moving the line in the x and y directions between the bounds of +/- d2
	// and changing the line's direction vector to point somewhere in its 3D "neighborhood" (inside a box with sides of 2*d around its original direction).
	float d = 0.15f;
	float d2 = 5.0f;
	NumericOptimizer< float, 6, CylinderMaximumFillFtor > optimizer;
	Vector< float, 6 > result = optimizer.optimize(
		Vector< float, 6 >( center[0]-d2, center[1]-d2, center[2], v1[0]-d, v1[1]-d, v1[2]-d ),
		Vector< float, 6 >( center[0]+d2, center[1]+d2, center[2], v1[0]+d, v1[1]+d, v1[2]+d ),
		Vector< float, 6 >( center[0], center[1], center[2], v1[0], v1[1], v1[2] ),
		CylinderMaximumFillFtor( aPoints )
		);

	center = result.GetSubVector< 0, 3 >();
	v1 = result.GetSubVector< 3, 6 >();
	LOG( "Result : " << result );
	LOG( "center : " << center );
	LOG( "v1 : " << v1 );

	VectorNormalization( v1 );

	//********************************************************************

	// Find the "endpoints" of the "axis" so it would exactly fit inside the smallest AABB containing the whole point-set.
	Vector3f intersection1;
	Vector3f intersection2;
	M4D::lineAABBIntersections( minimum, maximum,
			center, v1,
			intersection1, intersection2
			);

	// Set the direction vector of the "axis" to point toward its "upper" end.
	v1 = intersection2 - intersection1;
	VectorNormalization( v1 );

	// Find the distance of the nearest point from the point-set to the "axis".
	float minDistance = M4D::PointLineDistance( static_cast< Vector3f >( aPoints[0] ), center, v1 );
	for ( size_t i = 1; i < aPoints.size(); ++i ) {
		float tmp = M4D::PointLineDistance( static_cast< Vector3f >( aPoints[i] ), center, v1 );
		minDistance = M4D::min( minDistance, tmp );
	}

	// Get the length of the "axis" (fitting inside the smallest AABB containing the whole point-set).
	float height = VectorSize( intersection2 - intersection1 );

	// Fill in the findings to the result data structure.
	aProximalShaftMeasurementData.point = intersection1 - v1 * (0.1f * height);	// The lower end of the "axis" elongated by 10% of the axis' length.
	aProximalShaftMeasurementData.bboxP1 = intersection1;	// The lower end of the axis.
	aProximalShaftMeasurementData.bboxP2 = intersection2;	// The upper end of the axis.


	aProximalShaftMeasurementData.centerPoint = center;		// Center of the axis.
	aProximalShaftMeasurementData.direction = v1;			// Direction of the axis (pointing toward its upper end).
	aProximalShaftMeasurementData.height = height * 1.2;	// Length of the axis elongated by 20% (for 10% on the upper and the lower end respectively).
	aProximalShaftMeasurementData.radius = minDistance;		// Distance of the axis to the nearest point of the set.
	aProximalShaftMeasurementData.available = true;			// The data structure is filled with meaningful data.
	aProximalShaftMeasurementData.minimum = minimum;		// Minimum bounds to the point-set (the vertex of the minimal AABB to the set nearest to the origin).
	aProximalShaftMeasurementData.maximum = maximum;		// Maximum bounds to the point-set (the vertex of the minimal AABB to the set farthest from the origin).
}


