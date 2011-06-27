#ifndef SPHERE_H
#define SPHERE_H

#include "common/Vector.h"
#include "common/DefinitionMacros.h"

namespace M4D
{

template< unsigned taDim, typename TCoordType = float32 >
class Sphere
{
public:
	typedef Vector< TCoordType, taDim > PositionType;
	Sphere( const PositionType &aCenter = PositionType(), TCoordType aRadius = 0 ): mCenter( aCenter ), mRadius( aRadius )
	{}

	SIMPLE_GET_SET_METHODS( TCoordType, Radius, mRadius );
	SIMPLE_GET_SET_METHODS( PositionType, Center, mCenter );
	
	TCoordType &
	radius()
	{
		return mRadius;
	}
	
	const TCoordType &
	radius()const
	{
		return mRadius;
	}
	
	PositionType &
	center()
	{
		return mCenter;
	}
	
	const PositionType &
	center()const
	{
		return mCenter;
	}

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
	PositionType mCenter;
	TCoordType mRadius;
};

typedef Sphere< 2, float > Sphere2Df;
typedef Sphere< 2, double > Sphere2Dd;

typedef Sphere< 3, float > Sphere3Df;
typedef Sphere< 3, double > Sphere3Dd;

typedef Sphere< 2, float > Circlef;
typedef Sphere< 2, double > Circled;

} //M4D
#endif /*SPHERE_H*/
