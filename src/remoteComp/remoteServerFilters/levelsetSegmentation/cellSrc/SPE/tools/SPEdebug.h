#ifndef SPEDEBUG_H_
#define SPEDEBUG_H_

#include "stdio.h"

#ifdef DEBUG_LEVEL
/*
	#ifdef DEBUG_ADITIONAL_INFO
		#define D_PRINT( ARG )	\
			DOUT << __FILE__ \
			<< ":" << __LINE__ \
			<< ":" << ARG << std::endl;
	#else
	*/
		#define D_PRINT( ... )	\
			printf(__VA_ARGS__);
			//DOUT << ARG << std::endl;
//	#endif /*DEBUG_ADITIONAL_INFO*/

#else
#define	D_PRINT( ... )
#endif /*DEBUG_LEVEL*/

//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL
#define DL_PRINT( LEVEL, ... )	\
	if ( (DEBUG_LEVEL >= LEVEL) || (DEBUG_LEVEL == 0)){ \
				D_PRINT( __VA_ARGS__ );\
	}
#else
#define	DL_PRINT( LEVEL, ... )
#endif /*DEBUG_LEVEL*/

#endif /*SPEDEBUG_H_*/
