#include "GUI/utils/ViewConfiguration.h"
#include "common/MathTools.h"
#include "common/Log.h"

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector< float, 2 > &regionSize, const Vector< unsigned, 2 > &windowSize, ZoomType zoomType )
{	
	
	return GetOptimalViewConfiguration( Vector< float, 2 >( 0.0f ), regionSize, windowSize, zoomType );
}

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector< float, 2 > &regionMin, const Vector< float, 2 > &regionMax, const Vector< unsigned, 2 > &windowSize, ZoomType zoomType )
{	
	const Vector< float, 2 > regionSize( regionMax - regionMin );
	Vector< float, 2 > tmp( 
			regionSize[0] / static_cast< float >( windowSize[0] ), 
			regionSize[1] / static_cast< float >( windowSize[1] ) 
			);

	float aspectRatio = static_cast< float >(windowSize[0]) / static_cast< float >(windowSize[1]);
	float zoom = 1.0f;
	switch( zoomType ) {
	case ztFIT:
		zoom = Max( tmp[0], tmp[1] );
		break;
	case ztWIDTH_FIT:
		zoom = tmp[0];
		break;
	case ztHEIGHT_FIT:
		zoom = tmp[1];
		break;
	default:
		ASSERT( false );
	}

	float height = windowSize[1] * zoom;
	float width = windowSize[0] * zoom;

	Vector< float, 2 > offset;

	return ViewConfiguration2D( offset, height, aspectRatio );
}

