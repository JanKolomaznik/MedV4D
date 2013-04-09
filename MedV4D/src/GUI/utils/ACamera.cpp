#include "MedV4D/GUI/utils/ACamera.h"

void
ACamera::RotateAroundTarget( const Quaternion<ACamera::FloatType> &q )
{
	/*Position dist( mEyePos - mTargetPos );
	FloatType sizeSqr = dist * dist;*/


	Position direction = toGLM(RotatePoint(fromGLM(mTargetDirection), q));
	direction = glm::normalize(direction);
	Position dist = -mTargetDistance * direction;

	mEyePos = mTargetPos + dist;
	mUpDirection = toGLM(RotatePoint(fromGLM(mUpDirection), q ));
	mUpDirection = glm::normalize(mUpDirection);

	mTargetDirection = direction;
	mRightDirection = glm::cross(mTargetDirection, mUpDirection);
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
	glm::orthonormalize(mTargetDirection, mUpDirection);
	mUpDirection = glm::normalize(mUpDirection);

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
	glm::orthonormalize(mTargetDirection, mUpDirection);
	mUpDirection = glm::normalize(mUpDirection);

	UpdateRightDirection();
}

void
ACamera::SetUpDirection( const Direction & aUpDirection )
{
	mUpDirection = aUpDirection;
	glm::orthonormalize(mTargetDirection, mUpDirection);
	mUpDirection = glm::normalize(mUpDirection);

	UpdateRightDirection();
}

void
ACamera::YawAround( ACamera::FloatType angle )
{
	Quaternion<ACamera::FloatType> q = CreateRotationQuaternion( angle, fromGLM(mUpDirection) );
	RotateAroundTarget( q );
}

void
ACamera::PitchAround( ACamera::FloatType angle )
{
	Quaternion<ACamera::FloatType> q = CreateRotationQuaternion( angle, fromGLM(mRightDirection) );
	RotateAroundTarget( q );
}

void
ACamera::YawPitchAround( ACamera::FloatType yangle, ACamera::FloatType pangle )
{
	Quaternion<ACamera::FloatType> q = CreateRotationQuaternion( yangle, fromGLM(mUpDirection) ) * CreateRotationQuaternion( pangle, fromGLM(mRightDirection) );
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
