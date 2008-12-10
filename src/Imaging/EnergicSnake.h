#ifndef ENERGIC_SNAKE_H
#define ENERGIC_SNAKE_H

#include "Imaging/PointSet.h"
#include "Imaging/BSpline.h"

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file EnergicSnake.h 
 * @{ 
 **/

namespace Imaging
{
namespace Algorithms
{

template< typename ContourType >
class EFConvergeToPoint
{
public:
	typedef  M4D::Imaging::Geometry::PointSet< typename ContourType::Type, ContourType::Dimension > 	GradientType;
	typedef Coordinates< typename ContourType::Type, ContourType::Dimension >	PointCoordinate;

	void
	GetParametersGradient( ContourType &curve, GradientType &gradient )
	{
		if( curve.Size() != gradient.Size() ) {
			//TODO - solve problem
		}
		for( unsigned i = 0; i < gradient.Size(); ++i ) {
			gradient[i] = _point - curve[i];
			float32 size = sqrt(gradient[i]*gradient[i]);
			float32 pom = (size - 100.0f)/size;
			gradient[i] = pom * gradient[i];
		}
	}

	void
	SetCenterPoint( const PointCoordinate &point )
	{
		_point = point;
	}
private:
	PointCoordinate	_point;

};


template< typename ContourType, typename EnergyModel >
class EnergicSnake
{
public:
	typedef  M4D::Imaging::Geometry::PointSet< typename ContourType::Type, ContourType::Dimension > 	GradientType;

	EnergicSnake();

	~EnergicSnake();

	void
	Initialize( const ContourType & contour );

	uint32
	Step();

	bool
	Converge();

	bool
	Converged();

	void
	Reset();

	EnergyModel&
	GetEnergyModel()
		{ return _energyFunctional; }

	const ContourType&
	GetCurrentCurve()const
		{ return _curve; }

	const GradientType&
	GetCurrentGradient()const
		{ return _gradients[_actualGradient]; }

	const GradientType&
	GetPreviousGradient()const
		{ return _gradients[(_actualGradient+1)%2]; }
private:
	void
	SwitchGradients();
	void
	NormalizeGradient();
	float32
	GradientScalarProduct( const GradientType &v1, const GradientType &v2 );
	void
	ComputeCurveParametersGradient();
	void
	UpdateCurveParameters();
	void
	CheckSelfIntersection();
	void
	CheckSegmentLengths();
	void
	ComputeStepScale();

	EnergyModel	_energyFunctional;

	//
	GradientType	*_gradient;

	GradientType	_gradients[2];
	unsigned	_actualGradient;

	float32		_lastGradientSize;

	unsigned	_stepCount;


	//Parameters
	unsigned	_sampleRate;
	ContourType	_curve;
	float32		_stepScale;
	float32		_stepScaleAlpha;
	float32		_stepScaleBeta;
	float32		_minimalSegmentLength;
	float32		_maximalSegmentLength;
};

template< typename ContourType, typename EnergyModel >
EnergicSnake< ContourType, EnergyModel >
::EnergicSnake(): _gradient( NULL )
{
	_actualGradient = 1;

	_sampleRate = 5;

	_stepScale = 5.0;
	_stepScaleAlpha = 0.8;
	_stepScaleBeta = 0.5;
	_minimalSegmentLength = 10;
	_maximalSegmentLength = 30;

	SwitchGradients();
}

template< typename ContourType, typename EnergyModel >
EnergicSnake< ContourType, EnergyModel >
::~EnergicSnake()
{
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake< ContourType, EnergyModel >
::Initialize( const ContourType & contour )
{
	_curve = contour;
	_curve.Sample( _sampleRate );
}

template< typename ContourType, typename EnergyModel >
uint32
EnergicSnake< ContourType, EnergyModel >
::Step()
{
	++_stepCount;

	SwitchGradients();

	DL_PRINT(10, "EnergicSnake -> Step number " << _stepCount );
	ComputeCurveParametersGradient();

	DL_PRINT(10, "EnergicSnake ->    Update curve parameters " );
	UpdateCurveParameters();

	//Solve self intersection problem
	CheckSelfIntersection();

	//Divide or join segments with length out of tolerance
	CheckSegmentLengths();
	
	return _stepCount;
}

template< typename ContourType, typename EnergyModel >
bool
EnergicSnake< ContourType, EnergyModel >
::Converge()
{
	while( !Converged() ) {
		Step();
	}
	return true;
}

template< typename ContourType, typename EnergyModel >
bool
EnergicSnake< ContourType, EnergyModel >
::Converged()
{
	if( _stepCount > 1 ) {
		return _stepCount > 120 || _lastGradientSize < 0.00001;
	} else {
		return false;
	}
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake< ContourType, EnergyModel >
::Reset()
{

}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake< ContourType, EnergyModel >
::ComputeCurveParametersGradient()
{
	_gradient->Resize( _curve.Size() );

	_energyFunctional.GetParametersGradient( _curve, (*_gradient) );

	NormalizeGradient();
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake< ContourType, EnergyModel >
::UpdateCurveParameters()
{
	ComputeStepScale();

	for( size_t i=0; i < _curve.Size(); ++i ) {
		//std::cerr << _curve[i] << " : " ;
		_curve[i] += _stepScale * (*_gradient)[i];
		//std::cerr << _curve[i] << "\n";
	}
	_curve.Sample( _sampleRate );
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake< ContourType, EnergyModel >
::ComputeStepScale()
{
	if( _stepCount > 1 && _gradients[0].Size() == _gradients[1].Size() ) {
		float32 product = GradientScalarProduct( _gradients[0], _gradients[1] );
		_stepScale *= _stepScaleAlpha + _stepScaleBeta * product;

		DL_PRINT(15, "EnergicSnake ->      Compute step scale : " << _stepScale << " " << product );
	}
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake< ContourType, EnergyModel >
::CheckSelfIntersection()
{
	CoordInt2D seg = FindBSplineSelfIntersection( _curve );
	if( seg[0] < 0 ) return;
	
	DL_PRINT(10, "EnergicSnake ->    Found self intersections : " << seg[0] << "; " << seg[1] << " : " << _curve.GetSegmentCount() );
	unsigned inSegCount = static_cast< unsigned >( seg[1]-seg[0] );

	if( inSegCount == 0 ) {
		_curve.RemovePoint( seg[0] );
	} else {
		bool keepInInterval = inSegCount > (_curve.GetSegmentCount()-inSegCount);

		if( keepInInterval ) {
			DL_PRINT(10, "EnergicSnake ->     Removing interval : " << seg[1] << "; " << seg[0] );
			_curve.RemovePoints( seg[1], seg[0] );
		} else {
			DL_PRINT(10, "EnergicSnake ->     Removing interval : " << seg[0] << "; " << seg[1] );
			_curve.RemovePoints( seg[0], seg[1] );
		}
	}
	_curve.Sample( _sampleRate );
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake< ContourType, EnergyModel >
::CheckSegmentLengths()
{
	unsigned maxIdx;
	float32  maxVal;
	unsigned minIdx;
	float32  minVal;
	FindBSplineSegmentLengthExtremes( _curve, maxIdx, maxVal, minIdx, minVal );

	bool changed = false;
	if( maxVal > _maximalSegmentLength ) {
		_curve.SplitSegment( maxIdx );
		changed = true;
		DL_PRINT(10, "EnergicSnake ->     Spliting segment : " << maxIdx << " length = " << maxVal );
	}
	if( minVal < _minimalSegmentLength ) {
		_curve.JoinSegment( minIdx );
		changed = true;
		DL_PRINT(10, "EnergicSnake ->     Joining segment : " << minIdx << " length = " << minVal );
	}
	if( changed ) {
		_curve.Sample( _sampleRate );
	}
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake< ContourType, EnergyModel >
::SwitchGradients()
{
	_actualGradient = (_actualGradient + 1) % 2;

	_gradient = &(_gradients[_actualGradient]);
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake< ContourType, EnergyModel >
::NormalizeGradient()
{
	_lastGradientSize = GradientScalarProduct( (*_gradient), (*_gradient) );
	
	float32 tmp = 1.0f / sqrt( _lastGradientSize );

	for( size_t i=0; i < _gradient->Size(); ++i ) {
		(*_gradient)[i] *= tmp;
	}
}

template< typename ContourType, typename EnergyModel >
float32
EnergicSnake< ContourType, EnergyModel >
::GradientScalarProduct( 
		const EnergicSnake< ContourType, EnergyModel >::GradientType &v1, 
		const EnergicSnake< ContourType, EnergyModel >::GradientType &v2 )
{
	float32 product = 0.0;
	for( size_t i=0; i < _gradient->Size(); ++i ) {
		product += v1[i] * v2[i];
	}
	return product;
}

}/*namespace Algorithms*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/


#endif /*ENERGIC_SNAKE_H*/
