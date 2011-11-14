#include "ShoulderMeasurementModule/Computations.hpp"
#include "common/MathTools.h"
#include "common/GeometricAlgorithms.h"

#include <numeric>
#include <cstdlib>

#include <eigen2/Eigen/Dense>

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

	aHeadMeasurementData.point = center;
	aHeadMeasurementData.normal = normal;
	aHeadMeasurementData.vDirection = v1;
	aHeadMeasurementData.wDirection = v2;
	aHeadMeasurementData.available = true;
}


void
getProximalShaftMeasurementData( const PointSet &aPoints, ProximalShaftMeasurementData &aProximalShaftMeasurementData )
{
	Vector3f center;
	Vector3f v1;
	Eigen::Matrix3f covarianceMatrix;


	Vector3f minimum = aPoints[0];
	Vector3f maximum = aPoints[0];
	for ( size_t i = 1; i < aPoints.size(); ++i ) {
		minimum = M4D::min< float, 3 >( static_cast< const Vector3f &>( minimum ), static_cast< const Vector3f &>( aPoints[i] ) );
		maximum = M4D::max< float, 3 >( static_cast< const Vector3f &>( maximum ), static_cast< const Vector3f &>( aPoints[i] ) );
		/*D_PRINT( "point : " << aPoints[i] );
		D_PRINT( "min : " << minimum );
		D_PRINT( "max : " << maximum << "\n" );*/
	}

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


	Vector3f intersection1;
	Vector3f intersection2;
	M4D::lineAABBIntersections( minimum, maximum, 
			center, v1,
			intersection1, intersection2
			);
	
	v1 = intersection2 - intersection1;
	VectorNormalization( v1 );


	/*D_PRINT( "minT = " << minT );
	D_PRINT( "maxT = " << maxT );
	D_PRINT( "diffT = " << diffT );*/
	float minDistance = M4D::PointLineDistance( static_cast< Vector3f >( aPoints[0] ), center, v1 );
	for ( size_t i = 1; i < aPoints.size(); ++i ) {
		float tmp = M4D::PointLineDistance( static_cast< Vector3f >( aPoints[i] ), center, v1 );
		minDistance = M4D::min( minDistance, tmp );
	}

	float height = VectorSize( intersection2 - intersection1 );

	aProximalShaftMeasurementData.point = intersection1 - v1 * (0.1f * height);//center + minT * v1;
	aProximalShaftMeasurementData.bboxP1 = intersection1;
	aProximalShaftMeasurementData.bboxP2 = intersection2;


	aProximalShaftMeasurementData.centerPoint = center;
	aProximalShaftMeasurementData.direction = v1;
	aProximalShaftMeasurementData.height = height * 1.2;
	aProximalShaftMeasurementData.radius = minDistance;
	aProximalShaftMeasurementData.available = true;
	aProximalShaftMeasurementData.minimum = minimum;
	aProximalShaftMeasurementData.maximum = maximum;
}


