#ifndef DIRECTION_H
#define DIRECTION_H

#include "common/MathTools.h"
#include "common/Vector.h"
#include <cmath>

enum Direction { 
	dE	= 0, 
	dNE	= 1, 
	dN	= 2, 
	dNW	= 3, 
	dW	= 4, 
	dSW	= 5, 
	dS	= 6, 
	dSE	= 7   
};


inline Direction
OppositeDirection( Direction dir )
{
	return (Direction)((dir + 4) % 8);
}

inline Direction
DirectionForgetOrientation( Direction dir )
{
	return (Direction)(dir % 4);
}

template< typename T >
Direction
QuantizeDirectionDegree( T degree )
{
	if( degree < 0.0 || degree >= 360.0 ) {
		degree = degree - ((int)(degree / 360.0)) * 360.0; //TODO test
	}
	return (Direction)(ROUND(degree/ 45.0) % 8);
}

template< typename T >
Direction
QuantizeDirectionRadian( T radian )
{
	if( radian < 0.0 ) {
		radian = radian - ((int)(radian / PIx2) - 1) * PIx2; //TODO test
	}

	if( radian >= PIx2 ) {
		radian = radian - ((int)(radian / PIx2)) * PIx2; //TODO test
	}
	return (Direction)(ROUND(radian/ PId4) % 8);
}

template< typename T >
Direction
VectorDirection( const Vector< T, 2 > &v )
{
	return QuantizeDirectionRadian( atan2( v[1], v[0] ) );
}



#endif /*DIRECTION_H*/
