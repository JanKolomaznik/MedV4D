#include "MedV4D/GUI/utils/ACamera.h"

void
ACamera::rotateAroundTarget( const Quaternion<ACamera::FloatType> &q )
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
ACamera::setTargetPosition( const Position &pos )
{
	setTargetPosition( pos, mUpDirection );
}

void
ACamera::setTargetPosition( const Position &aPosition, const Position &aUpDirection )
{
	mTargetPos = aPosition;
	updateDistance();
	updateTargetDirection();

	mUpDirection = aUpDirection;
	glm::orthonormalize(mTargetDirection, mUpDirection);
	mUpDirection = glm::normalize(mUpDirection);

	updateRightDirection();
}

void
ACamera::setEyePosition( const Position &pos )
{
	setEyePosition( pos, mUpDirection );
}

void
ACamera::setEyePosition( const Position &aPosition, const Position &aUpDirection )
{
	mEyePos = aPosition;
	updateDistance();
	updateTargetDirection();

	mUpDirection = aUpDirection;
	glm::orthonormalize(mTargetDirection, mUpDirection);
	mUpDirection = glm::normalize(mUpDirection);

	updateRightDirection();
}

void
ACamera::setUpDirection( const Direction & aUpDirection )
{
	mUpDirection = aUpDirection;
	glm::orthonormalize(mTargetDirection, mUpDirection);
	mUpDirection = glm::normalize(mUpDirection);

	updateRightDirection();
}

void
ACamera::yawAround( ACamera::FloatType angle )
{
	Quaternion<ACamera::FloatType> q = CreateRotationQuaternion( angle, fromGLM(mUpDirection) );
	rotateAroundTarget( q );
}

void
ACamera::pitchAround( ACamera::FloatType angle )
{
	Quaternion<ACamera::FloatType> q = CreateRotationQuaternion( angle, fromGLM(mRightDirection) );
	rotateAroundTarget( q );
}

void
ACamera::yawPitchAround( ACamera::FloatType yangle, ACamera::FloatType pangle )
{
	Quaternion<ACamera::FloatType> q = CreateRotationQuaternion( yangle, fromGLM(mUpDirection) ) * CreateRotationQuaternion( pangle, fromGLM(mRightDirection) );
	rotateAroundTarget( q );
}

void
ACamera::yawPitchAbsolute( FloatType yangle, FloatType pangle )
{
	resetOrbit();
	yawPitchAround( yangle, pangle );
}

void
ACamera::resetOrbit()
{
	mEyePos = mTargetPos + Position( 0.0f, 0.0f, mTargetDistance );
	mUpDirection = Direction( 0.0f, 1.0f, 0.0f );
	updateTargetDirection();
	updateRightDirection();	
}

void
dollyCamera( ACamera &aCamera, float32 aRatio )
{
	ACamera::Position center = aCamera.targetPosition();
	ACamera::Position eye = aCamera.eyePosition();
	ACamera::Direction moveVector = (1.0f - aRatio) * (center - eye);
	
	aCamera.setEyePosition( eye + moveVector );
}
