#include "MedV4D/GUI/utils/ACamera.h"

void
ACamera::RotateAroundTarget( const Quaternion<ACamera::FloatType> &q )
{
	/*Position dist( mEyePos - mTargetPos );
	FloatType sizeSqr = dist * dist;*/


	Position direction = RotatePoint( mTargetDirection, q );
	VectorNormalization( direction );
	Position dist = -1.0f * direction * mTargetDistance;

	mEyePos = mTargetPos + dist;
	mUpDirection = RotatePoint( mUpDirection, q );
	VectorNormalization( mUpDirection );

	mTargetDirection = direction;
	mRightDirection = VectorProduct( mTargetDirection, mUpDirection );
}

void
ACamera::SetTargetPosition( const Position &pos )
{
	SetTargetPosition( pos, mUpDirection );
}

void
ACamera::SetTargetPosition( const Position &aPosition, const Position &aUpDirection )
{
	mTargetPos = aPosition;
	UpdateDistance();
	UpdateTargetDirection();

	mUpDirection = aUpDirection;
	Ortogonalize( mTargetDirection, mUpDirection );
	VectorNormalization( mUpDirection );

	UpdateRightDirection();
}

void
ACamera::SetEyePosition( const Position &pos )
{
	SetEyePosition( pos, mUpDirection );
}

void
ACamera::SetEyePosition( const Position &aPosition, const Position &aUpDirection )
{
	mEyePos = aPosition;
	UpdateDistance();
	UpdateTargetDirection();

	mUpDirection = aUpDirection;
	Ortogonalize( mTargetDirection, mUpDirection );
	VectorNormalization( mUpDirection );

	UpdateRightDirection();
}

void
ACamera::SetUpDirection( const Direction & aUpDirection )
{
	mUpDirection = aUpDirection;
	Ortogonalize( mTargetDirection, mUpDirection );
	VectorNormalization( mUpDirection );

	UpdateRightDirection();
}

void
ACamera::YawAround( ACamera::FloatType angle )
{
	Quaternion<ACamera::FloatType> q = CreateRotationQuaternion( angle, mUpDirection );
	RotateAroundTarget( q );
}

void
ACamera::PitchAround( ACamera::FloatType angle )
{
	Quaternion<ACamera::FloatType> q = CreateRotationQuaternion( angle, mRightDirection );
	RotateAroundTarget( q );
}

void
ACamera::YawPitchAround( ACamera::FloatType yangle, ACamera::FloatType pangle )
{
	Quaternion<ACamera::FloatType> q = CreateRotationQuaternion( yangle, mUpDirection ) * CreateRotationQuaternion( pangle, mRightDirection );
	RotateAroundTarget( q );
}

void
ACamera::YawPitchAbsolute( FloatType yangle, FloatType pangle )
{
	ResetOrbit();
	YawPitchAround( yangle, pangle );
}

void
ACamera::ResetOrbit()
{
	mEyePos = mTargetPos + Position( 0.0f, 0.0f, mTargetDistance );
	mUpDirection = Direction( 0.0f, 1.0f, 0.0f );
	UpdateTargetDirection();
	UpdateRightDirection();	
}

void
DollyCamera( ACamera &aCamera, float32 aRatio )
{
	ACamera::Position center = aCamera.GetTargetPosition();
	ACamera::Position eye = aCamera.GetEyePosition();
	ACamera::Direction moveVector = (1.0f - aRatio) * (center - eye);
	
	aCamera.SetEyePosition( eye + moveVector );
}
