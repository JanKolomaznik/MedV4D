#ifndef ENERGIC_SNAKE_H
#define ENERGIC_SNAKE_H

#include "MedV4D/Imaging/PointSet.h"
#include "MedV4D/Imaging/BSpline.h"
#include "MedV4D/Imaging/EnergyModels.h"

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

struct EnergicSnakeParameters
{
	unsigned	_selfIntersectionTestPeriod;
	unsigned	_segmentLengthsTestPeriod;
	unsigned	_sampleRate;
	float32		_minSegmentLength;
	float32		_maxSegmentLength;

	float32		_stepScale;
	float32		_stepScaleAlpha;
	float32		_stepScaleBeta;
	float32		_maxStepScale;

	EnergicSnakeParameters():
		_selfIntersectionTestPeriod( 2 ),
		_segmentLengthsTestPeriod( 2 ),
		_sampleRate( 5 ),
		_minSegmentLength( 3 ),
		_maxSegmentLength( 45 ),
		_stepScale( 5.0f ),
		_stepScaleAlpha( 0.95f ),
		_stepScaleBeta( 0.1f ),
		_maxStepScale( 100.0f )
	{
	}
};

struct EnergicSnakeStats
{

	float32		_lastGradientSize;

	unsigned	_stepCount;

	unsigned	_lastStructureChange;

	EnergicSnakeStats(): _lastGradientSize(0), _stepCount(0), _lastStructureChange(0)
		{}
};

class DefaultConvergenceCriterion
{
public:
	DefaultConvergenceCriterion(): _maxStepCount( 50 ), _calmDownInterval( 10 )
		{}
	bool
	Converged( EnergicSnakeParameters &params, EnergicSnakeStats &stats )
	{
		if( ((stats._stepCount >= _maxStepCount) && (stats._lastStructureChange >= _calmDownInterval)) 
			|| stats._stepCount >= 2*_maxStepCount ) {
			return true;
		}
		return false;
	}

	SIMPLE_GET_SET_METHODS( unsigned, MaxStepCount, _maxStepCount );
	SIMPLE_GET_SET_METHODS( unsigned, CalmDownInterval, _calmDownInterval );
private:
	unsigned _maxStepCount;
	unsigned _calmDownInterval;
};

class StepScaleConvergenceCriterion
{
public:
	StepScaleConvergenceCriterion(): _maxStepCount( 50 ), _calmDownInterval( 10 ), _stepScaleLimit( 0.01f )
		{}
	bool
	Converged( EnergicSnakeParameters &params, EnergicSnakeStats &stats )
	{
		if( (params._stepScale < _stepScaleLimit) && (stats._lastStructureChange >= _calmDownInterval) ) {
			return true;
		}
		if( (stats._stepCount >= _maxStepCount) && (stats._lastStructureChange >= _calmDownInterval) ) {
			return true;
		}
		if( stats._stepCount >= 2*_maxStepCount ) {
			return true;
		}
		return false;
	}

	SIMPLE_GET_SET_METHODS( unsigned, MaxStepCount, _maxStepCount );
	SIMPLE_GET_SET_METHODS( unsigned, CalmDownInterval, _calmDownInterval );
	SIMPLE_GET_SET_METHODS( float32, StepScaleLimit, _stepScaleLimit );
private:
	unsigned	_maxStepCount;
	unsigned	_calmDownInterval;
	float32		_stepScaleLimit;
};

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion = DefaultConvergenceCriterion >
class EnergicSnake: public EnergyModel, public ConvergenceCriterion
{
public:
	typedef  M4D::Imaging::Geometry::PointSet< typename ContourType::PointType > 	GradientType;

	EnergicSnake();

	~EnergicSnake();

	void
	Initialize( const ContourType & contour );

	int32
	Step();

	bool
	Converge();

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
		{ return _stats._lastGradientSize; }

	SIMPLE_GET_METHOD( unsigned, SampleRate, _parameters._sampleRate );
	void
	SetSampleRate( unsigned rate )
		{
			_curve.Sample( _parameters._sampleRate );
			_curve.SampleWithDerivations( _parameters._sampleRate );
		}

	SIMPLE_GET_SET_METHODS( unsigned, SelfIntersectionTestPeriod, _parameters._selfIntersectionTestPeriod );
	SIMPLE_GET_SET_METHODS( unsigned, SegmentLengthsTestPeriod, _parameters._segmentLengthsTestPeriod );
	SIMPLE_GET_SET_METHODS( float32, MinSegmentLength, _parameters._minSegmentLength );
	SIMPLE_GET_SET_METHODS( float32, MaxSegmentLength, _parameters._maxSegmentLength );
	SIMPLE_GET_SET_METHODS( float32, StepScale, _parameters._stepScale );
	SIMPLE_GET_SET_METHODS( float32, MaxStepScale, _parameters._maxStepScale );
	SIMPLE_GET_SET_METHODS( float32, StepScaleAlpha, _parameters._stepScaleAlpha );
	SIMPLE_GET_SET_METHODS( float32, StepScaleBeta, _parameters._stepScaleBeta );
protected:
	EnergicSnakeParameters	_parameters;
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

	EnergicSnakeStats	_stats;

	ContourType	_curve;
};

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::EnergicSnake(): _gradient( NULL )
{
	_actualGradient = 1;

	/*
	_selfIntersectionTestPeriod = 2;
	_segmentLengthsTestPeriod = 2;
	_sampleRate = 5;

	_stepScale = 5.0;
	_stepScaleAlpha = 0.95;
	_stepScaleBeta = 0.1;
	_maxStepScale = 100.0f;
	_minimalSegmentLength = 3;
	_maximalSegmentLength = 45;
	*/

	_stats._stepCount = 0;
	_stats._lastGradientSize = 0.0f;

	SwitchGradients();
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::~EnergicSnake()
{
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
void
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::Initialize( const ContourType & contour )
{
	this->ResetEnergy();

	_curve = contour;
	_curve.Sample( _parameters._sampleRate );
	_curve.SampleWithDerivations( _parameters._sampleRate );
	_stats._stepCount = 0;
	
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
int32
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::Step()
{
	try {
		++_stats._stepCount;
		SwitchGradients();

			DL_PRINT(10, "EnergicSnake -> Step number " << _stats._stepCount );
		ComputeCurveParametersGradient();

			DL_PRINT(10, "EnergicSnake ->    Update curve parameters " );
		UpdateCurveParameters();

		++_stats._lastStructureChange;
		if( _parameters._selfIntersectionTestPeriod && _stats._stepCount % _parameters._selfIntersectionTestPeriod == 0 ) 
		{
			_curve.Sample( _parameters._sampleRate );
			//Solve self intersection problem
			CheckSelfIntersection();
		}

		if( _parameters._segmentLengthsTestPeriod && _stats._stepCount % _parameters._segmentLengthsTestPeriod == 0 ) 
		{
			//Divide or join segments with length out of tolerance
			CheckSegmentLengths();
		}

		_curve.SampleWithDerivations( _parameters._sampleRate );
	} catch (ErrorHandling::ExceptionBase &e ) {
		LOG( "Exception thrown during optimization step " << _stats._stepCount << std::endl << e );
		_stats._lastGradientSize = 0.0f;
		return -1;
	}
	
	return _stats._stepCount;
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
bool
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::Converge()
{
	while( !this->Converged( _parameters, _stats ) ) {
		if( 0 > Step() ) {
			return false;
		}
	}
	return true;
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
void
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::Reset()
{
	this->ResetEnergy();
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
void
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::ComputeCurveParametersGradient()
{
	//Set gradient to same size as the curve has.
	_gradient->Resize( _curve.Size() );

	//Compute gradient
	_stats._lastGradientSize = this->GetParametersGradient( _curve, (*_gradient) );
	D_PRINT( "GRADIENT SIZE = " << _stats._lastGradientSize );

	//Normalize gradient to unit size ( 1/normalization factor )
	NormalizeGradient();
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
void
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::UpdateCurveParameters()
{
	ComputeStepScale();

	for( size_t i=0; i < _curve.Size(); ++i ) {
		_curve[i] += _parameters._stepScale * (*_gradient)[i];
	}
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
void
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::ComputeStepScale()
{
	if( _stats._stepCount > 1 && _gradients[0].Size() == _gradients[1].Size() ) {
		float32 product = GradientScalarProduct( _gradients[0], _gradients[1] );
		_parameters._stepScale *= _parameters._stepScaleAlpha + _parameters._stepScaleBeta * product;
		_parameters._stepScale = min( _parameters._maxStepScale, _parameters._stepScale );
		//_stepScale = 10;
		DL_PRINT(5, "EnergicSnake ->    Compute step scale : " << _parameters._stepScale << "; Product : " << product );
	}
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
void
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::CheckSelfIntersection()
{
	if( _curve.Size() <= 3 ) {
		_THROW_ ErrorHandling::ExceptionBase( "Self intersection test can't proceed - too few control points." );
	}
	CoordInt2D seg;
	if( !FindBSplineSelfIntersection( _curve, seg ) ) return;

	//We change spline topology - inform about that
	_stats._lastStructureChange = 0;

	DL_PRINT(10, "EnergicSnake ->    Found self intersections : " << seg[0] << "; " << seg[1] << " : " << _curve.GetSegmentCount() );
	if( (int32)(seg[0] + ContourType::Degree) >= seg[1] ) {
		_curve.RemovePoints( MOD(seg[1], _curve.Size()), MOD(seg[0] + ContourType::Degree + 1, _curve.Size()) );
		_curve.Sample( _parameters._sampleRate );
		return;
	}	
	unsigned inSegCount = seg[1]-(seg[0]+1);

	if( inSegCount < _curve.GetSegmentCount() - inSegCount - 2 ) {
		_curve.RemovePoints( MOD(seg[0]+2, _curve.Size()), MOD(seg[1]+2, _curve.Size()) );
	} else {
		_curve.RemovePoints( MOD(seg[1]+1, _curve.Size()), MOD(seg[0]+1, _curve.Size()) );
	}
	_curve.Sample( _parameters._sampleRate );
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
void
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::CheckSegmentLengths()
{
	unsigned maxIdx;
	float32  maxVal;
	unsigned minIdx;
	float32  minVal;

	FindBSplineSegmentLengthExtremes( _curve, maxIdx, maxVal, minIdx, minVal );

	bool changed = false;
	if( maxVal > _parameters._maxSegmentLength ) {
		DL_PRINT(10, "EnergicSnake ->     Spliting segment : " << maxIdx << " length = " << maxVal << " segCount = " << _curve.GetSegmentCount() );
		SplitSegment( _curve, maxIdx );
		changed = true;
		_curve.Sample( _parameters._sampleRate );
		//We change spline topology - inform about that
		_stats._lastStructureChange = 0;

		FindBSplineSegmentLengthExtremes( _curve, maxIdx, maxVal, minIdx, minVal );
	}
	if( _curve.Size() > 4 && minVal < _parameters._minSegmentLength ) {
		DL_PRINT(10, "EnergicSnake ->     Joining segments : " << minIdx << ", "<< minIdx << " length = " << minVal << " segCount = " << _curve.GetSegmentCount() );
		float32 prev = BSplineSegmentLength( _curve, MOD( minIdx-1, _curve.GetSegmentCount() ) );
		float32 next = BSplineSegmentLength( _curve, MOD( minIdx+1, _curve.GetSegmentCount() ) );
		if( prev < next ) {
			minIdx = MOD( minIdx-1, _curve.GetSegmentCount() );
		}
		JoinSegments( _curve, minIdx );
		changed = true;
		_curve.Sample( _parameters._sampleRate );

		//We change spline topology - inform about that
		_stats._lastStructureChange = 0;
	}
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
void
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::SwitchGradients()
{
	_actualGradient = (_actualGradient + 1) % 2;

	_gradient = &(_gradients[_actualGradient]);
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
void
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::NormalizeGradient()
{
	//_lastGradientSize = GradientScalarProduct( (*_gradient), (*_gradient) );
	float32 tmp = 0.0f;

	if( _stats._lastGradientSize < Epsilon ) {
		_stats._lastGradientSize = 0.0f;
	} else {
		//_lastGradientSize = sqrt( _lastGradientSize );
		tmp = 1.0f / _stats._lastGradientSize;
	}

	for( size_t i=0; i < _gradient->Size(); ++i ) {
		(*_gradient)[i] *= tmp;
	}
}

template< typename ContourType, typename EnergyModel, typename ConvergenceCriterion >
float32
EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >
::GradientScalarProduct( 
		const typename EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >::GradientType &v1, 
		const typename EnergicSnake< ContourType, EnergyModel, ConvergenceCriterion >::GradientType &v2 )
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
