#include "GUI/utils/ViewConfiguration.h"
#include "common/MathTools.h"
#include "common/Log.h"

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector< float, 2 > &regionSize, const Vector< unsigned, 2 > &windowSize, ZoomType zoomType )
{	
	Vector< float, 2 > tmp( static_cast< float >( windowSize[0] ) / regionSize[0], static_cast< float >( windowSize[1] ) / regionSize[1] );

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

	tmp = 0.5f * ( tmp - Vector< float, 2 >( zoom ) );
	Vector< float, 2 > offset = VectorMemberProduct( tmp, regionSize ); 

	return ViewConfiguration2D( offset, zoom, windowSize[0],  windowSize[1] );
}

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector< float, 2 > &regionMin, const Vector< float, 2 > &regionMax, const Vector< unsigned, 2 > &windowSize, ZoomType zoomType )
{	
	const Vector< float, 2 > regionSize( regionMax - regionMin );
	Vector< float, 2 > tmp( static_cast< float >( windowSize[0] ) / regionSize[0], static_cast< float >( windowSize[1] ) / regionSize[1] );
	Vector< float, 2 > tmp2( static_cast< float >( windowSize[0] ) / regionMax[0], static_cast< float >( windowSize[1] ) / regionMax[1] );

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

	tmp = 0.5f*( tmp - Vector< float, 2 >( zoom ) );
	Vector< float, 2 > offset = VectorMemberProduct( tmp, regionSize );

	return ViewConfiguration2D( offset, zoom);
}

