#ifndef _VIEW_CONFIGURATION_H
#define _VIEW_CONFIGURATION_H

#include "common/Vector.h"
#include "common/Log.h"
#include "common/Types.h"


struct ViewConfiguration2D
{
	ViewConfiguration2D(	
			Vector2f	aCenterPoint = Vector2f(),
			float32			aZoom = 1.0f,
			bool			aHFlip = false,
			bool			aVFlip = false
			): centerPoint( aCenterPoint ), zoom( aZoom ), hFlip( aHFlip ), vFlip( aVFlip )
	{
	}

	Vector< float32, 2 >	centerPoint;
	float32			zoom;
	bool			hFlip;
	bool			vFlip;



};

struct SliceViewConfig
{
	SliceViewConfig(): plane( XY_PLANE ), currentSlice( 0 )
	{}

	CartesianPlanes		plane;

	Vector< int32, 3 >	currentSlice;

	ViewConfiguration2D	viewConfiguration;
};



ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector2f &regionSize, const Vector< uint32, 2 > &windowSize, ZoomType zoomType = ztFIT );

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector2f &regionMin, const Vector2f &regionMax, const Vector< uint32, 2 > &windowSize, ZoomType zoomType = ztFIT );

Vector2f
GetRealCoordinatesFromScreen( const Vector2f &aScreenPos, const Vector< uint32, 2 > &windowSize, const ViewConfiguration2D &aConfig );

#endif /*_VIEW_CONFIGURATION_H*/
