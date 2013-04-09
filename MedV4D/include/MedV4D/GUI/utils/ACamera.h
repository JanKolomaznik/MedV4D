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
	eyePosition() const
		{ return mEyePos; }

	const Position &
	targetPosition() const
		{ return mTargetPos; }

	void
	setTargetPosition( const Position &pos );

	void
	setTargetPosition( const Position &aPosition, const Position &aUpDirection );

	void
	setEyePosition( const Position &pos );

	void
	setEyePosition( const Position &aPosition, const Position &aUpDirection );

	void
	setUpDirection( const Direction & );
		
	const Direction &
	upDirection() const
		{ return mUpDirection; }

	const Direction &
	targetDirection() const
		{ return mTargetDirection; }

	const Direction &
	rightDirection() const
		{ return mRightDirection; }

	FloatType
	targetDistance()
	{ return mTargetDistance; }

	/*SIMPLE_GET_SET_METHODS( FloatType, AspectRatio, mFieldOfViewY );
	SIMPLE_GET_SET_METHODS( FloatType, FieldOfView, mAspectRatio );
	SIMPLE_GET_SET_METHODS( FloatType, ZNear, mZNear );
	SIMPLE_GET_SET_METHODS( FloatType, ZFar, mZFar );*/

	void
	rotateAroundTarget( const Quaternion<FloatType> &q );

	void
	yawAround( FloatType angle );

	void
	pitchAround( FloatType angle );

	void
	yawPitchAround( FloatType yangle, FloatType pangle );

	void
	yawPitchAbsolute( FloatType yangle, FloatType pangle );

	
protected:
	void
	resetOrbit();

	void
	updateDistance()
	{ 
		mTargetDistance = glm::distance(mTargetPos, mEyePos);
	}
	void
	updateTargetDirection()
	{ 
		mTargetDirection = mTargetPos - mEyePos;
		mTargetDirection = glm::normalize(mTargetDirection);
	}
	void
	updateRightDirection()
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
	stream << "   Position : " << camera.eyePosition() << std::endl;
	stream << "   Target :   " << camera.targetPosition() << std::endl;
	stream << "   Up :       " << camera.upDirection() << std::endl;
	stream << "   Right :    " << camera.rightDirection() << std::endl;
	//stream << "   FOV :      " << camera.GetFieldOfView() << std::endl;
	return stream;
}

void
dollyCamera( ACamera &aCamera, float32 aRatio );


#endif /*ACAMERA_H*/

