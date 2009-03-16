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
	float32	inProbabilityPos;
	float32	outProbabilityPos;
	float32	logRationPos;
};

class ProbabilityGrid
{
public:
	typedef int32	IntensityType;
	typedef Vector< float32, 3 > Coordinates;

	ProbabilityGrid( Vector< float32, 3 > origin, Vector< uint32, 3 > gridSize ) :
		_originCoordiantes( origin ), _gridSize( _gridSize ), _strides( 1, gridSize[0], gridSize[0]*gridSize[1] )
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
	SIMPLE_GET_METHOD( Vector< uint32, 3 >, Size, _gridSize );
	SIMPLE_GET_METHOD( Vector< float32, 3 >, GridStep, _gridStep );
	SIMPLE_GET_METHOD( Vector< float32, 3 >, Origin, _originCoordiantes );
protected:

	Vector< uint32, 3 >
	GetClosestPoint( const Coordinates &pos )
	{
		Coordinates pom = pos - _originCoordiantes;
		return Vector< uint32, 3 >( 
				Round( pom[0]/_gridStep[0] ), 
				Round( pom[1]/_gridStep[1] ), 
				Round( pom[2]/_gridStep[2] )
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
	Histogram< float32 >	_inIntensity;
	Histogram< float32 >	_outIntensity;
	Histogram< float32 >	_logRatioIntensity;

	ProbabilityGrid		*_grid;
private:
}

}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*CANONICAL_PROB_MODEL_H*/
