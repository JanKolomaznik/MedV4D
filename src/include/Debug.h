#ifndef __DEBUG_H_
#define __DEBUG_H_

/**
 *  @ingroup common
 *  @file Debug.h
 *
 *  @addtogroup common
 *  @{
 *  @section DEBUG support
 *
 *  Defines some debuging tools that are (and should be) used all over
 *  the project.
 *
 * These symbols should be set at compilation time to turn on debugging prints.
 * DEBUG_LEVEL int
 * DEBUG_ADITIONAL_INFO
 */


#ifdef DEBUG_LEVEL
#include <iostream>
#include <string>
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
			DOUT <<"Assertion failed at " <<__FILE__ \
			<<", on line " <<__LINE__<<", in function \'" \
			<<__FUNCTION__<<"\'."<<std::endl; exit(1); }
#else
#define ASSERT(EXPR)
#endif /*DEBUG_LEVEL*/

//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL
	#ifdef DEBUG_ADITIONAL_INFO
		#define D_PRINT( ARG )	\
			DOUT << __FILE__ \
			<< ":" << __LINE__ \
			<< ":" << ARG << std::endl;
	#else
		#define D_PRINT( ARG )	\
			DOUT << ARG << std::endl;
	#endif /*DEBUG_ADITIONAL_INFO*/

#else
#define	D_PRINT(ARG)
#endif /*DEBUG_LEVEL*/

//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL	
#define D_PRINT_NOENDL( ARG )	\
	DOUT << ARG;
#else
#define	D_PRINT_NOENDL(ARG)
#endif /*DEBUG_LEVEL*/

//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL
#define DL_PRINT( LEVEL, ARG )	\
	if ( (DEBUG_LEVEL >= LEVEL) || (DEBUG_LEVEL == 0)){ \
				D_PRINT( ARG );\
	}
#else
#define	DL_PRINT( LEVEL, ARG)
#endif /*DEBUG_LEVEL*/

//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL
#define D_COMMAND( ARG )	ARG
#else
#define	D_COMMAND(ARG)
#endif /*DEBUG_LEVEL*/

//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL
#define DL_COMMAND( LEVEL, ARG )	\
	if ( (DEBUG_LEVEL >= LEVEL) || (DEBUG_LEVEL == 0)){ \
				D_COMMAND( ARG );\
	}
#else
#define	DL_COMMAND(ARG)
#endif /*DEBUG_LEVEL*/

//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL
class DebugCommentObject
{
public:
	DebugCommentObject( std::string enter, std::string leave )
		: _leaveText( leave )
	{
		D_PRINT( enter );
	}

	~DebugCommentObject()
	{
		D_PRINT( _leaveText );
	}
private:
	std::string	_leaveText;
};
#endif /*DEBUG_LEVEL*/

#ifdef DEBUG_LEVEL
#define D_BLOCK_COMMENT( ENTER_TEXT, LEAVE_TEXT ) DebugCommentObject ____DEBUG_BLOCK_OBJ##__LINE__ = DebugCommentObject( ENTER_TEXT, LEAVE_TEXT );
#else
#define D_BLOCK_COMMENT( ENTER_TEXT, LEAVE_TEXT )
#endif /*DEBUG_LEVEL*/
//----------------------------------------------------------------------------

/**
 * Value used in exception debug prints.
 **/
#define EXCEPTION_DEBUG_LEVEL	10

//----------------------------------------------------------------------------

/** @} */

#endif /*__DEBUG_H_*/