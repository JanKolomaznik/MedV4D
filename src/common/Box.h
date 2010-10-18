#ifndef BOX_H
#define BOX_H

#include "common/Vector.h"
#include "common/DefinitionMacros.h"

template< unsigned taDim, typename TCoordType = float32 >
class Box
{
public:
	typedef Vector< TCoordType, taDim > PositionType;
	Box( const PositionType &aFirstCorner, const PositionType &aSecondCorner ): mFirstCorner( Min( aFirstCorner, aSecondCorner ) ), mSecondCorner( Max( aFirstCorner, aSecondCorner ) )
	{}

protected:
	PositionType aFirstCorner;
	PositionType aSecondCorner;
};


#endif /*BOX_H*/
