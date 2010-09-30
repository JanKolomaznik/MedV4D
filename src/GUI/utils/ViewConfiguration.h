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

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector< float32, 2 > &regionSize, const Vector< uint32, 2 > &windowSize, ZoomType zoomType = ztFIT );

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector< float32, 2 > &regionMin, const Vector< float32, 2 > &regionMax, const Vector< uint32, 2 > &windowSize, ZoomType zoomType = ztFIT );

#endif /*_VIEW_CONFIGURATION_H*/
