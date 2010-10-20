#include "GUI/utils/Camera.h"

void
Camera::RotateAroundCenter( const Quaternion<Camera::FloatType> &q )
{
	Position dist( _eyePos - _centerPos );
	FloatType sizeSqr = dist * dist;

	dist = RotatePoint( dist, q );
	VectorNormalization( dist );
	dist *= Sqrt( sizeSqr );

	_eyePos = _centerPos + dist;
	_upDirection = RotatePoint( _upDirection, q );
	VectorNormalization( _upDirection );

	_centerDirection = RotatePoint( _centerDirection, q );
	VectorNormalization( _centerDirection );

	_rightDirection = VectorProduct( _centerDirection, _upDirection );
}

void
Camera::SetCenterPosition( const Position &pos )
{
	_centerPos = pos;
	_centerDirection = _centerPos - _eyePos;
	VectorNormalization( _centerDirection );
	
	Ortogonalize( _centerDirection, _upDirection );
	VectorNormalization( _upDirection );
	_rightDirection = VectorProduct( _centerDirection, _upDirection );
}

void
Camera::SetEyePosition( const Position &pos )
{
	_eyePos = pos;
	_centerDirection = _centerPos - _eyePos;
	VectorNormalization( _centerDirection );
	
	Ortogonalize( _centerDirection, _upDirection );
	VectorNormalization( _upDirection );
	_rightDirection = VectorProduct( _centerDirection, _upDirection );
}

void
Camera::YawAround( Camera::FloatType angle )
{
	Quaternion<Camera::FloatType> q = CreateRotationQuaternion( angle, _upDirection );
	RotateAroundCenter( q );
}

void
Camera::PitchAround( Camera::FloatType angle )
{
	Quaternion<Camera::FloatType> q = CreateRotationQuaternion( angle, _rightDirection );
	RotateAroundCenter( q );
}

void
Camera::YawPitchAround( Camera::FloatType yangle, Camera::FloatType pangle )
{
	Quaternion<Camera::FloatType> q = CreateRotationQuaternion( yangle, _upDirection ) * CreateRotationQuaternion( pangle, _rightDirection );
	RotateAroundCenter( q );
}

void
DollyCamera( Camera &aCamera, float32 aRatio )
{
	Camera::Position center = aCamera.GetCenterPosition();
	Camera::Position eye = aCamera.GetEyePosition();
	Camera::Direction moveVector = (1.0f - aRatio) * (center - eye);
	
	aCamera.SetEyePosition( eye + moveVector );
}
