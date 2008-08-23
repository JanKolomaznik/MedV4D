#ifndef COMMON_H
#define COMMON_H

/**
 * our defines saying what OS we are building for
 * they are connected to well known symbols
 */
#define OS_WIN
//#define OS_LINUX
//#define OS_BSD
//#define OS_MAC

#include "Types.h"
#include "Debug.h"
#include "Log.h"
#include "ExceptionBase.h"
#include <iomanip>

//*****************************************************************************
/**
 * Macro for easy generic programing - mixing static and dynamic polymorhism.
 * Create switch command with cases over dimension numbers.
 * @param SWITCH Statement which will be placed in "switch( SWITCH )".
 * @param ... Templated command. Must contain DIM, which will be replaced by 
 * apropriete value. Example : function_name< DIM >( DIM + 1 )
 **/
#define DIMENSION_TEMPLATE_SWITCH_MACRO( SWITCH, ... ) \
	switch( SWITCH ) {\
	case 2:{ const unsigned DIM = 2; __VA_ARGS__ ; } break;\
	case 3:{ const unsigned DIM = 3; __VA_ARGS__ ; } break;\
	default: ASSERT( false );\
	}
	//TODO
	/*case 4:{ const unsigned DIM = 4; __VA_ARGS__ ; } break;\*/


//*****************************************************************************
template< unsigned Dim1, unsigned Dim2 >
struct IsSameDimension;

template< unsigned Dim >
struct IsSameDimension< Dim, Dim >
{
	//Only possible is when both parameters are the same value.
};
//*****************************************************************************



//TODO - move
#define PROHIBIT_COPYING_OF_OBJECT_MACRO( ClassName ) \
	ClassName( const ClassName& ); \
	ClassName& \
	operator=( const ClassName& ); 



//TODO test and move to better place
#define MAKESTRING( S ) #S

using namespace M4D::ErrorHandling;

#endif
