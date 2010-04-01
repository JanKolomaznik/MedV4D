#ifndef __DEBUG_H_
#define __DEBUG_H_

/**
 *  @ingroup common
 *  @file Debug.h
 *
 *  @addtogroup common
 *  @{
 *  @section debug DEBUG support
 *
 *  First thing which should be considered before starting big project is
 *  how it will be debugged. We have set of preprocessor macros, which
 *  allow us to use debug prints, conditional compilation, debuging prints
 *  for code blocks. 
 *
 *  Some of these macros have version, which takes numeric parameter - 
 *  debugging level. Actual debugging level is passed to compiler in 
 *  command line as value assigned to preprocessor definition: 
 *  DEBUG_LEVEL= int. This enables debugging and set debugging level. 
 *  If you in addition pass to compiler definition DEBUG_ADITIONAL_INFO
 *  some of mentioned debugging tools will have richer output.
 */


#ifdef DEBUG_LEVEL
#include <iostream>
#include <string>
#include <iomanip>

extern std::string	____EXCEPTION_FILE_NAME;
extern int		____EXCEPTION_LINE_NUMBER;

void
ResetExceptionInfo();
/**
 * Do not use pdout explicitely in your code.
 */ 
extern std::ostream *pdout;
#define DOUT	(*pdout)

#endif /*DEBUG_LEVEL*/


#ifdef DEBUG_LEVEL
#define _THROW_	____EXCEPTION_FILE_NAME = __FILE__; \
		____EXCEPTION_LINE_NUMBER = __LINE__; \
		throw
#else
#define _THROW_ throw
#endif /*DEBUG_LEVEL*/


//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL
#define SET_DOUT( N_OSTREAM )	(pdout = &(N_OSTREAM))
#else
#define SET_DOUT( N_OSTREAM )
#endif /*DEBUG_LEVEL*/


//----------------------------------------------------------------------------
#ifdef DEBUG_LEVEL
#define ASSERT_INFO(EXPR, info) if ( (EXPR) == 0 ) { \
			DOUT <<"Assertion failed at " <<__FILE__ \
			<<", on line " <<__LINE__<<", in function \'" \
			<<__FUNCTION__<<"\'. Reason: " << info <<std::endl; exit(1); }
#else
#define ASSERT_INFO(EXPR, info)
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
	#define DS_PRINT( ARG )	\
		DOUT << ARG << std::endl;
#else
#define	DS_PRINT(ARG)
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
#define D_COMMAND( ... )	__VA_ARGS__
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
	DebugCommentObject( std::string enter, std::string leave, std::string file )
		: _leaveText( leave ), _file( file )
	{
		DS_PRINT( _file << ": " << enter );
	}

	~DebugCommentObject()
	{
		DS_PRINT( _file << ": " << _leaveText );
	}
private:
	std::string	_leaveText;
	std::string	_file;
};
#endif /*DEBUG_LEVEL*/

#ifdef DEBUG_LEVEL
#define D_BLOCK_COMMENT( ENTER_TEXT, LEAVE_TEXT ) DebugCommentObject ____DEBUG_BLOCK_OBJ##__LINE__ = DebugCommentObject( ENTER_TEXT, LEAVE_TEXT, __FILE__ );
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

