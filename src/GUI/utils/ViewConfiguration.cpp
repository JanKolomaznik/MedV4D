#include "MedV4D/GUI/utils/ViewConfiguration.h"
#include "MedV4D/Common/MathTools.h"
#include "MedV4D/Common/Log.h"
#include <algorithm>

namespace M4D
{

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector< float32, 2 > &regionSize, const Vector< uint32, 2 > &windowSize, ZoomType zoomType )
{	
	
	return GetOptimalViewConfiguration( Vector< float, 2 >( 0.0f ), regionSize, windowSize, zoomType );
}

ViewConfiguration2D 
GetOptimalViewConfiguration( const Vector2f &regionMin, const Vector2f &regionMax, const Vector< uint32, 2 > &windowSize, ZoomType zoomType )
{	
	const Vector2f regionSize( regionMax - regionMin );
	Vector< float, 2 > tmp( 
			static_cast< float >( windowSize[0] ) / regionSize[0], 
			static_cast< float >( windowSize[1] ) / regionSize[1] 
			);

	float32 zoom = 1.0f;
	switch( zoomType ) {
	case ztFIT:
		zoom = min( tmp[0], tmp[1] );
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

Vector2f
GetRealCoordinatesFromScreen( const Vector2f &aScreenPos, const Vector< uint32, 2 > &windowSize, const ViewConfiguration2D &aConfig )
{
	Vector2f hsize = ( 0.5f / aConfig.zoom ) *  Vector2f( windowSize[0], windowSize[1] );
	Vector2f min =  aConfig.centerPoint - hsize;
	Vector2f max =  aConfig.centerPoint + hsize;

	Vector2f ratio = Vector2f( aScreenPos[0] / windowSize[0], aScreenPos[1] / windowSize[1] );
	if ( aConfig.hFlip ) {
		std::swap( min[0], max[0] );
	}
	if ( aConfig.vFlip ) {
		std::swap( min[1], max[1] );
	}

	return min + VectorMemberProduct( ratio, max - min );
}


}/*namespace M4D*/
