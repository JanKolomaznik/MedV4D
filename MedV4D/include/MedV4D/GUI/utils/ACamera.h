#ifndef ACAMERA_H
#define ACAMERA_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/Vector.h"
#include "MedV4D/Common/Quaternion.h"
#include <glm/glm.hpp>
#include <glm/ext.hpp>

inline std::ostream &
operator<< (std::ostream &out, const glm::vec3 &vec) {
    out << "[" 
        << vec.x << " " << vec.y << " "<< vec.z 
        << "]";

    return out;
}

inline glm::fvec3
vectorProjection(const glm::fvec3 &u, const glm::fvec3 &v)
{
	return glm::dot(v,u) * u;
}

inline glm::fvec3
toGLM(const Vector3f &aVec)
{
	return glm::fvec3(aVec[0], aVec[1], aVec[2]);
}

inline Vector3f
fromGLM(const glm::fvec3 &aVec)
{
	return Vector3f(aVec.x, aVec.y, aVec.z);
}

class ACamera
{
public:
	typedef float	FloatType;
	typedef /*Vector< FloatType, 3 >*/glm::fvec3 Position;
	typedef /*Vector< FloatType, 3 >*/glm::fvec3 Direction;

	ACamera( const Position &eye = Position(), const Position &center = Position() ) 
		: mTargetPos( center ), mEyePos( eye ), mUpDirection( 0.0f, 1.0f, 0.0f ), mTargetDirection( center - eye ), mRightDirection( 1.0f, 0.0f, 0.0f )
		/*mFieldOfViewY( 45.0f ), mAspectRatio( 1.0f ), mZNear( 0.5f ), mZFar( 10000 )*/
	{
		mTargetDirection = glm::normalize(mTargetDirection);
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

	void
	SetUpDirection( const Direction & );
		
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

	/*SIMPLE_GET_SET_METHODS( FloatType, AspectRatio, mFieldOfViewY );
	SIMPLE_GET_SET_METHODS( FloatType, FieldOfView, mAspectRatio );
	SIMPLE_GET_SET_METHODS( FloatType, ZNear, mZNear );
	SIMPLE_GET_SET_METHODS( FloatType, ZFar, mZFar );*/

	void
	RotateAroundTarget( const Quaternion<FloatType> &q );

	void
	YawAround( FloatType angle );

	void
	PitchAround( FloatType angle );

	void
	YawPitchAround( FloatType yangle, FloatType pangle );

	void
	YawPitchAbsolute( FloatType yangle, FloatType pangle );

	
protected:
	void
	ResetOrbit();

	void
	UpdateDistance()
	{ 
		mTargetDistance = glm::distance(mTargetPos, mEyePos);
	}
	void
	UpdateTargetDirection()
	{ 
		mTargetDirection = mTargetPos - mEyePos;
		mTargetDirection = glm::normalize(mTargetDirection);
	}
	void
	UpdateRightDirection()
	{ 
		mRightDirection = glm::cross(mTargetDirection, mUpDirection);
	}
	
	//Quaternion<FloatType>	mRotation;

	Position		mTargetPos;
	Position		mEyePos;

	//All normalized
	Direction		mUpDirection;
	Direction		mTargetDirection;
	Direction		mRightDirection;

 	FloatType  		mTargetDistance; 

	/*FloatType  		mFieldOfViewY; 
 	FloatType  		mAspectRatio; 
 	FloatType  		mZNear; 
 	FloatType  		mZFar;*/
};

inline std::ostream &
operator<<( std::ostream & stream, ACamera & camera )
{
	stream << "Camera info" << std::endl;
	stream << "   Position : " << camera.GetEyePosition() << std::endl;
	stream << "   Target :   " << camera.GetTargetPosition() << std::endl;
	stream << "   Up :       " << camera.GetUpDirection() << std::endl;
	stream << "   Right :    " << camera.GetRightDirection() << std::endl;
	//stream << "   FOV :      " << camera.GetFieldOfView() << std::endl;
	return stream;
}

void
DollyCamera( ACamera &aCamera, float32 aRatio );


#endif /*ACAMERA_H*/

