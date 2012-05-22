#include "MedV4D/Imaging/Image.h"
#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/MathTools.h"
#include <boost/static_assert.hpp>
#include <set>

namespace M4D {
namespace Imaging {
namespace painting {


template< typename TImage, typename TBrush >
void
paintLineWithBrush( TImage &aImage, Vector<float, TImage::Dimension> aStartPoint, Vector<float, TImage::Dimension> aEndPoint, TBrush &aBrush )
{
	typename TImage::ElementExtentsType eExtents = aImage.GetElementExtents();
	typedef Vector<float,TImage::Dimension> Vect;
	
	Vect dir = aEndPoint - aStartPoint;
	float dst = VectorSize( dir );
	VectorNormalization( dir );
	
	float delta = M4D::min( eExtents );
	Vect current = aStartPoint;
	aBrush.apply( aStartPoint );
	for( size_t i = 1; i < dst / delta; ++i ) {
		
		aBrush.apply( current );
		current += dir * delta;
	}
	aBrush.apply( aEndPoint );
	
}


template< typename TImage >
void
paintLineWithBrush( TImage &aImage, Vector<float, TImage::Dimension> aStartPoint, Vector<float, TImage::Dimension> aEndPoint, float aRadius, CartesianPlanes aPlane )
{

	
}

template< typename TImage >
void
drawRectangleAlongLine( TImage &aImage, typename TImage::Element aValue, Vector3f aStartPoint, Vector3f aEndPoint, float aWidth, Vector3f aPlaneNormal )
{
	BOOST_STATIC_ASSERT( TImage::Dimension == 3 );
	
	Vector3i c1, c2, c3, c4, lMin, lMax, coord, tmp( 1, 1, 1 );
	Vector3f dir = aEndPoint - aStartPoint;
	VectorNormalization( dir );
	Vector3f binormal = VectorProduct( dir, aPlaneNormal );
	VectorNormalization( binormal );
	
	c1 = aImage.GetElementCoordsFromWorldCoords( aStartPoint + aWidth * binormal );
	c2 = aImage.GetElementCoordsFromWorldCoords( aEndPoint + aWidth * binormal );
	c3 = aImage.GetElementCoordsFromWorldCoords( aEndPoint - aWidth * binormal );
	c4 = aImage.GetElementCoordsFromWorldCoords( aStartPoint - aWidth * binormal );
	
	lMin = M4D::minVect<int,3>( c1, c2, c3, c4 );
	lMax = M4D::maxVect<int,3>( c1, c2, c3, c4 );
	lMin -= tmp;
	lMax += tmp;
	lMin = M4D::maxVect<int,3>( lMin, aImage.GetMinimum() );
	lMax = M4D::minVect<int,3>( lMax, aImage.GetMaximum() );
	
	for( coord[2] = lMin[2]; coord[2] <= lMax[2]; ++coord[2] ) {
		for( coord[1] = lMin[1]; coord[1] <= lMax[1]; ++coord[1] ) {
			for( coord[0] = lMin[0]; coord[0] <= lMax[0]; ++coord[0] ) {
				aImage.GetElement( coord ) = aValue;
			}
		}
	}
}

template< typename TImage >
void
getValuesFromRectangleAlongLine( const TImage &aImage, std::set< typename TImage::Element > &aValues, Vector3f aStartPoint, Vector3f aEndPoint, float aWidth, Vector3f aPlaneNormal )
{
	BOOST_STATIC_ASSERT( TImage::Dimension == 3 );
	
	Vector3i c1, c2, c3, c4, lMin, lMax, coord, tmp( 1, 1, 1 );
	Vector3f dir = aEndPoint - aStartPoint;
	VectorNormalization( dir );
	Vector3f binormal = VectorProduct( dir, aPlaneNormal );
	VectorNormalization( binormal );
	
	c1 = aImage.GetElementCoordsFromWorldCoords( aStartPoint + aWidth * binormal );
	c2 = aImage.GetElementCoordsFromWorldCoords( aEndPoint + aWidth * binormal );
	c3 = aImage.GetElementCoordsFromWorldCoords( aEndPoint - aWidth * binormal );
	c4 = aImage.GetElementCoordsFromWorldCoords( aStartPoint - aWidth * binormal );
	
	lMin = M4D::minVect<int,3>( c1, c2, c3, c4 );
	lMax = M4D::maxVect<int,3>( c1, c2, c3, c4 );
	lMin -= tmp;
	lMax += tmp;
	lMin = M4D::maxVect<int,3>( lMin, aImage.GetMinimum() );
	lMax = M4D::minVect<int,3>( lMax, aImage.GetMaximum() );
	
	for( coord[2] = lMin[2]; coord[2] <= lMax[2]; ++coord[2] ) {
		for( coord[1] = lMin[1]; coord[1] <= lMax[1]; ++coord[1] ) {
			for( coord[0] = lMin[0]; coord[0] <= lMax[0]; ++coord[0] ) {
				aValues.insert( aImage.GetElement( coord ) );
			}
		}
	}
}


} //namespace M4D
} //namespace Imaging
} //namespace painting