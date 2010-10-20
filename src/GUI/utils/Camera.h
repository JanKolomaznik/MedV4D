#ifndef CAMERA_H
#define CAMERA_H

#include "common/Common.h"
#include "common/Vector.h"
#include "common/Quaternion.h"


class Camera
{
public:
	typedef float	FloatType;
	typedef Vector< FloatType, 3 > Position;
	typedef Vector< FloatType, 3 > Direction;

	Camera( const Position &eye = Position(), const Position &center = Position() ) 
		: _centerPos( center ), _eyePos( eye ), _upDirection( 0.0f, 1.0f, 0.0f ), _centerDirection( center - eye ), _rightDirection( 1.0f, 0.0f, 0.0f ),
		_fieldOfViewY( 45.0f ), _aspectRatio( 1.0f ), _zNear( 0.5f ), _zFar( 10000 )
	{
		VectorNormalization( _centerDirection );
	}

	void
	Reset();

	const Position &
	GetEyePosition() const
		{ return _eyePos; }

	const Position &
	GetCenterPosition() const
		{ return _centerPos; }

	void
	SetCenterPosition( const Position &pos );

	void
	SetEyePosition( const Position &pos );

	const Direction &
	GetUpDirection() const
		{ return _upDirection; }

	const Direction &
	GetCenterDirection() const
		{ return _centerDirection; }

	const Direction &
	GetRightDirection() const
		{ return _rightDirection; }

	SIMPLE_GET_SET_METHODS( FloatType, AspectRatio, _aspectRatio );
	SIMPLE_GET_SET_METHODS( FloatType, FieldOfView, _fieldOfViewY );
	SIMPLE_GET_SET_METHODS( FloatType, ZNear, _zNear );
	SIMPLE_GET_SET_METHODS( FloatType, ZFar, _zFar );

	void
	RotateAroundCenter( const Quaternion<FloatType> &q );

	void
	YawAround( FloatType angle );

	void
	PitchAround( FloatType angle );

	void
	YawPitchAround( FloatType yangle, FloatType pangle );
protected:
	
	
	Quaternion<FloatType>	_rotation;

	Position		_centerPos;
	Position		_eyePos;

	//All normalized
	Direction		_upDirection;
	Direction		_centerDirection;
	Direction		_rightDirection;

	//FloatType		_distance;

	FloatType  		_fieldOfViewY; 
 	FloatType  		_aspectRatio; 
 	FloatType  		_zNear; 
 	FloatType  		_zFar;
};


void
DollyCamera( Camera &aCamera, float32 aRatio );


#endif /*CAMERA_H*/

