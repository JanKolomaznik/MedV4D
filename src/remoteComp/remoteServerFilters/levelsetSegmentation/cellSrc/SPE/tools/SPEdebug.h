#ifndef SPEDEBUG_H_
#define SPEDEBUG_H_

#include "stdio.h"

//#define SPE_DEBUG_TO_FILE 1

#if defined(DEBUG_LEVEL) && defined(SPE_DEBUG_TO_FILE)
extern FILE *debugFile;
#endif

#ifdef DEBUG_LEVEL
#ifdef SPE_DEBUG_TO_FILE
		#define D_PRINT( ... )	\
			fprintf(debugFile, __VA_ARGS__);
#else
		#define D_PRINT( ... )	\
			printf(__VA_ARGS__);
#endif	/*SPE_DEBUG_TO_FILE*/
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

#ifdef DEBUG_LEVEL
#define D_COMMAND( ARG )	ARG
#else
#define	D_COMMAND(ARG)
#endif /*DEBUG_LEVEL*/

#endif /*SPEDEBUG_H_*/
