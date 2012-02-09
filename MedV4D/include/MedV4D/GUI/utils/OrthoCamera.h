#ifndef ORTHO_CAMERA_H
#define ORTHO_CAMERA_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/Vector.h"
#include "MedV4D/Common/Quaternion.h"

#include "MedV4D/GUI/utils/ACamera.h"


class OrthoCamera: public ACamera
{
public:
	typedef float	FloatType;
	typedef Vector< FloatType, 3 > Position;
	typedef Vector< FloatType, 3 > Direction;
	
	OrthoCamera( const Position &eye = Position(), const Position &center = Position() ) 
		: ACamera( eye, center ), mLeft( -1.0f ), mRight( 1.0f ), mBottom( -1.0f ), mTop( 1.0f ), mNear( 0.0f ), mFar( 1000.0f )
	{ }
	
	SIMPLE_GET_SET_METHODS( FloatType, Left, mLeft );
	SIMPLE_GET_SET_METHODS( FloatType, Right, mRight );
	SIMPLE_GET_SET_METHODS( FloatType, Bottom, mBottom );
	SIMPLE_GET_SET_METHODS( FloatType, Top, mTop );
	SIMPLE_GET_SET_METHODS( FloatType, Near, mNear );
	SIMPLE_GET_SET_METHODS( FloatType, Far, mFar );
	
	void
	SetWindow( FloatType aWidth, FloatType aHeight )
	{
		mLeft = -0.5f*aWidth;
		mRight = 0.5f*aWidth;
		mTop = 0.5f*aHeight;
		mBottom = -0.5f*aHeight;
	}
protected:
	FloatType mLeft, mRight, mBottom, mTop, mNear, mFar;
};

#endif /*ORTHO_CAMERA_H*/