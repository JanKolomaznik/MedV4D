#ifndef _VIEW_CONFIGURATION_H
#define _VIEW_CONFIGURATION_H

#include "common/Vector.h"
#include "common/Log.h"

enum ZoomType
{
	ztFIT,
	ztWIDTH_FIT,
	ztHEIGHT_FIT
};

struct ViewConfiguration2D
{
	ViewConfiguration2D(	
			Vector< float, 2 >	_offset = Vector< float, 2 >(),
			float			_zoom = 1.0f,
			float			_width = 1.0f,
			float			_height = 1.0f,
			int			_hFlip = 1,
			int			_vFlip = 1
			): offset( _offset ), zoom( _zoom ), hFlip( _hFlip ), vFlip( _vFlip ), width( _width ), height( _height )
	{
	}

	Vector< float32, 2 >	offset;
	float32			zoom;
	int			hFlip;
	int			vFlip;
	float			width;
	float			height;
};

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector< float32, 2 > &regionSize, const Vector< unsigned, 2 > &windowSize, ZoomType zoomType = ztFIT );

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector< float, 2 > &regionMin, const Vector< float, 2 > &regionMax, const Vector< unsigned, 2 > &windowSize, ZoomType zoomType = ztFIT );
#endif /*_VIEW_CONFIGURATION_H*/
