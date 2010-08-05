#include "GUI/utils/ViewConfiguration.h"
#include "common/MathTools.h"
#include "common/Log.h"

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector< float32, 2 > &regionSize, const Vector< uint32, 2 > &windowSize, ZoomType zoomType )
{	
	
	return GetOptimalViewConfiguration( Vector< float, 2 >( 0.0f ), regionSize, windowSize, zoomType );
}

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector< float32, 2 > &regionMin, const Vector< float32, 2 > &regionMax, const Vector< uint32, 2 > &windowSize, ZoomType zoomType )
{	
	const Vector< float, 2 > regionSize( regionMax - regionMin );
	Vector< float, 2 > tmp( 
			static_cast< float >( windowSize[0] ) / regionSize[0], 
			static_cast< float >( windowSize[1] ) / regionSize[1] 
			);

	float32 zoom = 1.0f;
	switch( zoomType ) {
	case ztFIT:
		zoom = Min( tmp[0], tmp[1] );
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

	return ViewConfiguration2D( regionMin + (0.5f * regionSize), zoom );
}

