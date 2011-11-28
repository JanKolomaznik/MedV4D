#ifndef BOX_H
#define BOX_H

#include "MedV4D/Common/Vector.h"
#include "MedV4D/Common/DefinitionMacros.h"

template< unsigned taDim, typename TCoordType = float32 >
class AABox
{
public:
	typedef Vector< TCoordType, taDim > PositionType;
	AABox( const PositionType &aFirstCorner, const PositionType &aSecondCorner ): mFirstCorner( min( aFirstCorner, aSecondCorner ) ), mSecondCorner( max( aFirstCorner, aSecondCorner ) )
	{}

protected:
	PositionType aFirstCorner;
	PositionType aSecondCorner;
};

/*template< unsigned taDim, typename TCoordType = float32 >
class Box
{
public:
	typedef Vector< TCoordType, taDim > PositionType;
	Box( const PositionType &aFirstCorner, const PositionType &aSecondCorner ): mFirstCorner( min( aFirstCorner, aSecondCorner ) ), mSecondCorner( max( aFirstCorner, aSecondCorner ) )
	{}

protected:
	PositionType aFirstCorner;
	PositionType aSecondCorner;
};*/

#endif /*BOX_H*/
