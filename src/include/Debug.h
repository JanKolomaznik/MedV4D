#ifndef __DEBUG_H_
#define __DEBUG_H_

/**
 * These symbols should be set at compilation time to turn on debuging prints.
 * DEBUG_LEVEL int
 * DEBUG_ADITIONAL_INFO
 */


#ifdef DEBUG_LEVEL
#include <iomanip>
/**
 * Do not use pdout explicitely in your code.
 */ 
extern std::ostream *pdout;
#define DOUT	(*pdout)

#endif /*DEBUG_LEVEL*/




//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL
#define SET_DOUT( N_OSTREAM )	(pdout = &(N_OSTREAM))
#else
#define SET_DOUT( N_OSTREAM )
#endif /*DEBUG_LEVEL*/


//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL
#define ASSERT(EXPR) 	if ( (EXPR) == 0 ) { \
			DOUT <<"Asserted at " <<__FILE__ \
			<<", on line " <<__LINE__<<", in function " \
			<<__FUNCTION__<<std::endl; exit; }
#else
#define ASSERT(EXPR)
#endif /*DEBUG_LEVEL*/

//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL
#define D_PRINT( ARG )	\
	#ifdef DEBUG_ADITIONAL_INFO\
		DOUT <<"File: " <<__FILE__ \
		<<"; Line: " <<__LINE__\
		<<"\t"<<ARG;\
	#else\
		DOUT <<ARG;\
	#endif /*DEBUG_ADITIONAL_INFO*/

#else
#define	D_PRINT(ARG)
#endif /*DEBUG_LEVEL*/

#ifdef DEBUG_LEVEL
#define DL_PRINT( LEVEL, ARG )	\
	#if ( (DEBUG_LEVEL >= LEVEL) || (DEBUG_LEVEL == 0)) \
				D_PRINT( ARG );\
	#endif /*(DEBUG_LEVEL >= LEVEL) || (DEBUG_LEVEL == 0)*/
#else
#define	DL_PRINT(ARG)
#endif /*DEBUG_LEVEL*/

//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL
#define D_COMMAND( ARG )	(ARG)
#else
#define	D_COMMAND(ARG)
#endif /*DEBUG_LEVEL*/

#ifdef DEBUG_LEVEL
#define DL_COMMAND( LEVEL, ARG )	\
	#if ( (DEBUG_LEVEL >= LEVEL) || (DEBUG_LEVEL == 0)) \
				D_COMMAND( ARG );\
	#endif /*(DEBUG_LEVEL >= LEVEL) || (DEBUG_LEVEL == 0)*/
#else
#define	DL_COMMAND(ARG)
#endif /*DEBUG_LEVEL*/


//----------------------------------------------------------------------------

#endif /*__DEBUG_H_*/
