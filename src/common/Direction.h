#ifndef DIRECTION_H
#define DIRECTION_H

#include "common/MathTools.h"
#include "common/Vector.h"

enum Direction { 
	dE, dNE, dN, dNW, dW, dSW, dS, dSE   
};


inline Direction
OpossiteDirection( Direction dir )
{
	return (Direction)((dir + 4) % 8);
}

template< typename T >
Direction
QuantizeDirectionDegree( T degree )
{
	if( degree < 0.0 || degree >= 360.0 ) {
		degree = degree - ((int)(degree / 360.0)) * 360.0; //TODO test
	}
	return ROUND(degree/ 45.0) % 8;
}

template< typename T >
Direction
QuantizeDirectionRadian( T radian )
{
	if( radian < 0.0 || radian >= PIx2 ) {
		radian = radian - ((int)(radian / PIx2)) * PIx2; //TODO test
	}
	return ROUND(radian/ PId4) % 8;
}



#endif /*DIRECTION_H*/
