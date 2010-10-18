#ifndef SPHERE_H
#define SPHERE_H

#include "common/Vector.h"
#include "common/DefinitionMacros.h"

template< unsigned taDim, typename TCoordType = float32 >
class Sphere
{
public:
	typedef Vector< TCoordType, taDim > PositionType;
	Sphere( const PositionType &aCenter, TCoordType aRadius ): mCenter( aCenter ), mRadius( aRadius )
	{}

	SIMPLE_GET_SET_METHODS( TCoordType, Radius, mRadius );
	SIMPLE_GET_SET_METHODS( PositionType, Center, mCenter );

	/*void
	Merge( const Sphere &aSphere )
	{
		TCoordType distance = VectorDistance( mCenter, aSphere.mCenter );

		if ( mRadius < aSphere.mRadius ) {
			if ( (distance + mRadius) <= aSphere.mRadius ) {
				mRadius = aSphere.mRadius;
				mCenter = aSphere.mCenter;
				return;
			}
		} else {
			if ( (distance + aSphere.mRadius) < mRadius ) {
				return;
			}
		}
		TCoordType diameter = distance + mRadius + aSphere.mRadius;
		TCoordType tmp = diameter / 2 - mRadius;
		
		mCenter = (tmp / distance) * mCenter + (1 - (tmp / distance)) *aSphere.mCenter;
		mRadius = diameter / 2;
	}*/
protected:
};


#endif /*SPHERE_H*/
