#ifndef ENERGIC_SNAKE_H
#define ENERGIC_SNAKE_H

#include "Imaging/PointSet.h"
#include "Imaging/BSpline.h"
#include "Imaging/EnergyModels.h"

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

template< typename ContourType, typename EnergyModel >
class EnergicSnake: public EnergyModel
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

	const ContourType&
	GetCurrentCurve()const
		{ return _curve; }

	const GradientType&
	GetCurrentGradient()const
		{ return _gradients[_actualGradient]; }

	const GradientType&
	GetPreviousGradient()const
		{ return _gradients[(_actualGradient+1)%2]; }

	unsigned
	GetCurrentGradientSize()const
		{ return _lastGradientSize; }

	unsigned
	GetSelfIntersectionTestPeriod()const
		{ return _selfIntersectionTestPeriod; }

	void
	SetSelfIntersectionTestPeriod( unsigned period )
		{ _selfIntersectionTestPeriod = period; }

	unsigned
	GetSegmentLengthsTestPeriod()const
		{ return _segmentLengthsTestPeriod; }

	void
	SetSegmentLengthsTestPeriod( unsigned period )
		{ _segmentLengthsTestPeriod = period; }

	unsigned
	GetSampleRate()const
		{ return _sampleRate; }

	void
	SetSampleRate( unsigned rate )
		{ _sampleRate = rate; }

	float32
	GetMinSegmentLength()const
		{ return _minimalSegmentLength; }

	void
	SetMinSegmentLength( float32 len )
		{ _minimalSegmentLength = len; }

	float32
	GetMaxSegmentLength()const
		{ return _maximalSegmentLength; }

	void
	SetMaxSegmentLength( float32 len )
		{ _maximalSegmentLength = len; }
protected:
	//Parameters
	unsigned	_selfIntersectionTestPeriod;
	unsigned	_segmentLengthsTestPeriod;
	unsigned	_sampleRate;
	float32		_minimalSegmentLength;
	float32		_maximalSegmentLength;

	float32		_stepScale;
	float32		_stepScaleAlpha;
	float32		_stepScaleBeta;
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

	//
	GradientType	*_gradient;

	GradientType	_gradients[2];
	unsigned	_actualGradient;

	float32		_lastGradientSize;

	unsigned	_stepCount;


	ContourType	_curve;
};

template< typename ContourType, typename EnergyModel >
EnergicSnake< ContourType, EnergyModel >
::EnergicSnake(): _gradient( NULL )
{
	_actualGradient = 1;

	_selfIntersectionTestPeriod = 2;
	_segmentLengthsTestPeriod = 2;
	_sampleRate = 5;

	_stepScale = 5.0;
	_stepScaleAlpha = 0.95;
	_stepScaleBeta = 0.1;
	_minimalSegmentLength = 3;
	_maximalSegmentLength = 45;

	_stepCount = 0;
	_lastGradientSize = 0.0f;

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
	this->ResetEnergy();

	_curve = contour;
	_curve.Sample( _sampleRate );
	_curve.SampleWithDerivations( _sampleRate );
	_stepCount = 0;
	
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

	if( _stepCount % _selfIntersectionTestPeriod == 0 ) 
	{
		_curve.Sample( _sampleRate );
		//Solve self intersection problem
	//	CheckSelfIntersection();
	}

	if( _stepCount % _segmentLengthsTestPeriod == 0 ) 
	{
		//Divide or join segments with length out of tolerance
	//	CheckSegmentLengths();
	}

	_curve.ReSampleWithDerivations();
	
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
	if( _stepCount > 10 ) {
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
	this->ResetEnergy();
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake< ContourType, EnergyModel >
::ComputeCurveParametersGradient()
{
	//Set gradient to same size as the curve has.
	_gradient->Resize( _curve.Size() );

	try {
		//Compute gradient
		_lastGradientSize = this->GetParametersGradient( _curve, (*_gradient) );
		D_PRINT( "GRADIENT SIZE = " << _lastGradientSize );

		//Normalize gradient to unit size ( 1/normalization factor )
		NormalizeGradient();
	} catch (ErrorHandling::ExceptionBase &e ) {
		LOG( "Exception thrown during optimization step " << _stepCount << std::endl << e );
		_lastGradientSize = 0.0f;
	}
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake< ContourType, EnergyModel >
::UpdateCurveParameters()
{
	ComputeStepScale();

	for( size_t i=0; i < _curve.Size(); ++i ) {
		_curve[i] += _stepScale * (*_gradient)[i];
	}
}

template< typename ContourType, typename EnergyModel >
void
EnergicSnake< ContourType, EnergyModel >
::ComputeStepScale()
{
	if( _stepCount > 1 && _gradients[0].Size() == _gradients[1].Size() ) {
		float32 product = GradientScalarProduct( _gradients[0], _gradients[1] );
		_stepScale *= _stepScaleAlpha + _stepScaleBeta * product;

		//_stepScale = 10;
		DL_PRINT(5, "EnergicSnake ->      Compute step scale : " << _stepScale << " " << product );
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
	//_lastGradientSize = GradientScalarProduct( (*_gradient), (*_gradient) );
	float32 tmp = 0.0f;

	if( _lastGradientSize < Epsilon ) {
		_lastGradientSize = 0.0f;
	} else {
		//_lastGradientSize = sqrt( _lastGradientSize );
		tmp = 1.0f / _lastGradientSize;
	}

	for( size_t i=0; i < _gradient->Size(); ++i ) {
		(*_gradient)[i] *= tmp;
	}
}

template< typename ContourType, typename EnergyModel >
float32
EnergicSnake< ContourType, EnergyModel >
::GradientScalarProduct( 
		const typename EnergicSnake< ContourType, EnergyModel >::GradientType &v1, 
		const typename EnergicSnake< ContourType, EnergyModel >::GradientType &v2 )
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
