#ifndef COMMON_H
#define COMMON_H

/**
 *  @defgroup common Common tools
 */

/**
 *  @ingroup common
 *  @file Common.h
 *
 *  @addtogroup common
 *  @{
 *  
 *  Commons contains commonly used code as well as includes of others
 *  commonly used code. In code then only Commons.h is included.
 */

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
#include "Endianess.h"
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


#define Max( a, b ) ((a)<(b) ? (b) : (a))
#define Min( a, b ) ((a)<(b) ? (a) : (b))
#define MOD( a, b ) ((a)<0 ? ((a)%(b)) + (b) : (a) % (b))
#define PWR( a ) ( (a) * (a) )
#define ROUND( a ) ( (int)(a+0.5) )

template< typename NType >
inline NType
Abs( NType a ) {
	if( (a)<0 ) return -1 * a;

	return a;
}

template< typename NType >
inline NType
Sqr( NType a ) {
	return a*a;
}

template< typename NType >
inline int32
Sgn( NType a ) {
	if( a < 0 ) {
		return -1;
	} 
	if( a > 0 ) {
		return 1;
	} 

	return 0;
}

extern const float32 Epsilon;
extern const float32 PI;

//***********************************************************

//TODO - move
#define PROHIBIT_COPYING_OF_OBJECT_MACRO( ClassName ) \
	ClassName( const ClassName& ); \
	ClassName& \
	operator=( const ClassName& ); 


//TODO test and move to better place
#define MAKESTRING( S ) #S

using namespace M4D::ErrorHandling;

/** @} */

#endif
