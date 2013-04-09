#ifndef CAMERA_H
#define CAMERA_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/Vector.h"
#include "MedV4D/Common/Quaternion.h"

#include "MedV4D/GUI/utils/ACamera.h"

class Camera: public ACamera
{
public:
	typedef ACamera::FloatType FloatType;
	typedef ACamera::Position Position;
	typedef ACamera::Direction Direction;

	Camera( const Position &eye = Position(), const Position &center = Position() ) 
		: ACamera( eye, center ), mFieldOfViewY( 45.0f ), mAspectRatio( 1.0f ), mZNear( 0.5f ), mZFar( 10000 )
	{ }

	SIMPLE_GET_SET_METHODS( FloatType, AspectRatio, mFieldOfViewY );
	SIMPLE_GET_SET_METHODS( FloatType, FieldOfView, mAspectRatio );
	SIMPLE_GET_SET_METHODS( FloatType, ZNear, mZNear );
	SIMPLE_GET_SET_METHODS( FloatType, ZFar, mZFar );


protected:
	FloatType  		mFieldOfViewY; 
 	FloatType  		mAspectRatio; 
 	FloatType  		mZNear; 
 	FloatType  		mZFar;
};

inline std::ostream &
operator<<( std::ostream & stream, Camera & camera )
{
	stream << "Camera info" << std::endl;
	stream << "   Position : " << camera.eyePosition() << std::endl;
	stream << "   Target :   " << camera.targetPosition() << std::endl;
	stream << "   Up :       " << camera.upDirection() << std::endl;
	stream << "   Right :    " << camera.rightDirection() << std::endl;
	stream << "   FOV :      " << camera.GetFieldOfView() << std::endl;
	return stream;
}

/*void
DollyCamera( Camera &aCamera, float32 aRatio );*/


#endif /*CAMERA_H*/

