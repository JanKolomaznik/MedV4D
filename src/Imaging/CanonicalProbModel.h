#ifndef CANONICAL_PROB_MODEL_H
#define CANONICAL_PROB_MODEL_H

#include "Common.h"
#include <cmath>
#include "Imaging/Histogram.h"

namespace M4D
{
namespace Imaging
{


struct GridPointRecord
{
	GridPointRecord() 
		: inProbabilityPos( 0 ),
		outProbabilityPos( 0 ),
		logRationPos( 0.0 )
	{}
	GridPointRecord( 
		float32	pinProbabilityPos,
		float32	poutProbabilityPos,
		float32	plogRationPos
		) 
		: inProbabilityPos( pinProbabilityPos ),
		outProbabilityPos( poutProbabilityPos ),
		logRationPos( plogRationPos )
	{}
	float32	inProbabilityPos;
	float32	outProbabilityPos;
	float32	logRationPos;
};

class ProbabilityGrid
{
public:
	typedef int32	IntensityType;
	typedef Vector< float32, 3 > Coordinates;

	typedef Vector< float32, 3 > Vector3F;
	typedef Vector< uint32, 3 > Vector3UI;

	ProbabilityGrid( Vector< float32, 3 > origin, Vector< uint32, 3 > gridSize, Vector< float32, 3 > step ) :
		_gridStep( step ), _originCoordiantes( origin ), _gridSize( _gridSize ), _strides( 1, gridSize[0], gridSize[0]*gridSize[1] )
	{
		_grid = new GridPointRecord[ _gridSize[0]*_gridSize[1]*_gridSize[2] ];
	}

	~ProbabilityGrid()
	{
		delete [] _grid;
	}

	//***********************************************************************
	float32
	InProbabilityPosition( const Coordinates &pos )
	{
		return GetPointRecord( GetClosestPoint( pos ) ).inProbabilityPos;
	}

	float32
	OutProbabilityPosition( const Coordinates &pos )
	{
		return GetPointRecord( GetClosestPoint( pos ) ).outProbabilityPos;
	}

	float32
	LogRatioProbabilityPosition( const Coordinates &pos )
	{
		return GetPointRecord( GetClosestPoint( pos ) ).logRationPos;
	}
	//***********************************************************************
	GridPointRecord &
	GetPointRecord( const Vector< uint32, 3 > &pos )
	{
		if( !(pos < _gridSize) ) {
			return _outlier;
		}
		return _grid[ _strides * pos ];
	}

	const GridPointRecord &
	GetPointRecord( const Vector< uint32, 3 > &pos )const
	{
		if( !(pos < _gridSize) ) {
			return _outlier;
		}
		return _grid[ _strides * pos ];
	}
	SIMPLE_GET_METHOD( Vector3UI, Size, _gridSize );
	SIMPLE_GET_METHOD( Vector3F, GridStep, _gridStep );
	SIMPLE_GET_METHOD( Vector3F, Origin, _originCoordiantes );
protected:

	Vector< uint32, 3 >
	GetClosestPoint( const Coordinates &pos )
	{
		Coordinates pom = pos - _originCoordiantes;
		return Vector< uint32, 3 >( 
				ROUND( pom[0]/_gridStep[0] ), 
				ROUND( pom[1]/_gridStep[1] ), 
				ROUND( pom[2]/_gridStep[2] )
				);
	}


	Vector< float32, 3 >	_gridStep;

	Vector< float32, 3 >	_originCoordiantes;

	Vector< uint32, 3 >	_gridSize;
	Vector< uint32, 3 >	_strides;

	GridPointRecord		*_grid;//< Array of grid point records

	GridPointRecord		_outlier;
};

class CanonicalProbModel
{
public:
	typedef int32	IntensityType;
	typedef Vector< float32, 3 > Coordinates;

	//***********************************************************************
	float32
	InProbabilityIntesity( IntensityType intensity );

	float32
	OutProbabilityIntesity( IntensityType intensity );
	
	float32
	LogRatioProbabilityIntesity( IntensityType intensity );
	//***********************************************************************

	//***********************************************************************
	float32
	InProbabilityIntesityPosition( IntensityType intensity, const Coordinates &pos );
	
	float32
	OutProbabilityIntesityPosition( IntensityType intensity, const Coordinates &pos );
	
	float32
	LogRatioProbabilityIntesityPosition( IntensityType intensity, const Coordinates &pos );
	//***********************************************************************

	//***********************************************************************
	float32
	InProbabilityPosition( const Coordinates &pos );

	float32
	OutProbabilityPosition( const Coordinates &pos );

	float32
	LogRatioProbabilityPosition( const Coordinates &pos );
	//***********************************************************************
protected:
	CanonicalProbModel( ProbabilityGrid *grid, int32 interestMin, int32 interestMax ) : 
		_inIntensity( interestMin, interestMax ),
		_outIntensity( interestMin, interestMax ),
		_logRatioIntensity( interestMin, interestMax ),
		_grid( grid )
		{}

	Histogram< float32 >	_inIntensity;
	Histogram< float32 >	_outIntensity;
	Histogram< float32 >	_logRatioIntensity;

	ProbabilityGrid		*_grid;
private:
};

}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*CANONICAL_PROB_MODEL_H*/
