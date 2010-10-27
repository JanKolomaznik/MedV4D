#include "GUI/utils/Camera.h"

void
Camera::RotateAroundTarget( const Quaternion<Camera::FloatType> &q )
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

	/*D_PRINT("CAMERA PARAMETERS ------------------------------" );
	D_PRINT(mTargetPos);
	D_PRINT(mEyePos);

	//All normalized
	D_PRINT(mUpDirection);
	D_PRINT(mTargetDirection);
	D_PRINT(mRightDirection);

 	D_PRINT(mTargetDistance); 

	D_PRINT(mFieldOfViewY); 
 	D_PRINT(mAspectRatio); 
 	D_PRINT(mZNear); 
 	D_PRINT(mZFar);
	D_PRINT("CAMERA PARAMETERS ------------------------------" );*/
}

void
Camera::SetTargetPosition( const Position &pos )
{
	SetTargetPosition( pos, mUpDirection );
}

void
Camera::SetTargetPosition( const Position &aPosition, const Position &aUpDirection )
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
Camera::SetEyePosition( const Position &pos )
{
	SetEyePosition( pos, mUpDirection );
}

void
Camera::SetEyePosition( const Position &aPosition, const Position &aUpDirection )
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
Camera::YawAround( Camera::FloatType angle )
{
	Quaternion<Camera::FloatType> q = CreateRotationQuaternion( angle, mUpDirection );
	RotateAroundTarget( q );
}

void
Camera::PitchAround( Camera::FloatType angle )
{
	Quaternion<Camera::FloatType> q = CreateRotationQuaternion( angle, mRightDirection );
	RotateAroundTarget( q );
}

void
Camera::YawPitchAround( Camera::FloatType yangle, Camera::FloatType pangle )
{
	Quaternion<Camera::FloatType> q = CreateRotationQuaternion( yangle, mUpDirection ) * CreateRotationQuaternion( pangle, mRightDirection );
	RotateAroundTarget( q );
}

void
DollyCamera( Camera &aCamera, float32 aRatio )
{
	Camera::Position center = aCamera.GetTargetPosition();
	Camera::Position eye = aCamera.GetEyePosition();
	Camera::Direction moveVector = (1.0f - aRatio) * (center - eye);
	
	aCamera.SetEyePosition( eye + moveVector );
}
