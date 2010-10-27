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
		: mTargetPos( center ), mEyePos( eye ), mUpDirection( 0.0f, 1.0f, 0.0f ), mTargetDirection( center - eye ), mRightDirection( 1.0f, 0.0f, 0.0f ),
		mFieldOfViewY( 45.0f ), mAspectRatio( 1.0f ), mZNear( 0.5f ), mZFar( 10000 )
	{
		VectorNormalization( mTargetDirection );
	}

	void
	Reset();

	const Position &
	GetEyePosition() const
		{ return mEyePos; }

	const Position &
	GetTargetPosition() const
		{ return mTargetPos; }

	void
	SetTargetPosition( const Position &pos );

	void
	SetTargetPosition( const Position &aPosition, const Position &aUpDirection );

	void
	SetEyePosition( const Position &pos );

	void
	SetEyePosition( const Position &aPosition, const Position &aUpDirection );

	const Direction &
	GetUpDirection() const
		{ return mUpDirection; }

	const Direction &
	GetTargetDirection() const
		{ return mTargetDirection; }

	const Direction &
	GetRightDirection() const
		{ return mRightDirection; }

	FloatType
	GetTargetDistance()
	{ return mTargetDistance; }

	SIMPLE_GET_SET_METHODS( FloatType, AspectRatio, mFieldOfViewY );
	SIMPLE_GET_SET_METHODS( FloatType, FieldOfView, mAspectRatio );
	SIMPLE_GET_SET_METHODS( FloatType, ZNear, mZNear );
	SIMPLE_GET_SET_METHODS( FloatType, ZFar, mZFar );

	void
	RotateAroundTarget( const Quaternion<FloatType> &q );

	void
	YawAround( FloatType angle );

	void
	PitchAround( FloatType angle );

	void
	YawPitchAround( FloatType yangle, FloatType pangle );
protected:
	
	void
	UpdateDistance()
	{ 
		mTargetDistance = VectorDistance( mTargetPos, mEyePos );
	}
	void
	UpdateTargetDirection()
	{ 
		mTargetDirection = mTargetPos - mEyePos;
		VectorNormalization( mTargetDirection );
	}
	void
	UpdateRightDirection()
	{ 
		mRightDirection = VectorProduct( mTargetDirection, mUpDirection );
	}
	
	Quaternion<FloatType>	mRotation;

	Position		mTargetPos;
	Position		mEyePos;

	//All normalized
	Direction		mUpDirection;
	Direction		mTargetDirection;
	Direction		mRightDirection;

 	FloatType  		mTargetDistance; 

	FloatType  		mFieldOfViewY; 
 	FloatType  		mAspectRatio; 
 	FloatType  		mZNear; 
 	FloatType  		mZFar;
};


void
DollyCamera( Camera &aCamera, float32 aRatio );


#endif /*CAMERA_H*/

