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
#include <sstream>

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
	//Only possible when both parameters are the same value.
};
//*****************************************************************************


//#define Max( a, b ) ((a)<(b) ? (b) : (a))
//#define Min( a, b ) ((a)<(b) ? (a) : (b))
//#define MOD( a, b ) ((a)<0 ? ((a)%(b)) + (b) : (a) % (b))
#define PWR( a ) ( (a) * (a) )
#define ROUND( a ) ( (int)(a+0.5) )

template< typename NTypeA, typename NTypeB >
NTypeB
MOD( NTypeA a, NTypeB b );

template<>
inline int32
MOD( int32 a, int32 b )
{
	int32 val = a % b;
	if( val < 0 ) {
		val += b;
	}
	return val;
}

template<>
inline uint32
MOD( int32 a, uint32 b )
{
	int32 val = a % b;
	if( val < 0 ) {
		val += b;
	}
	return (uint32)val;
}

template<>
inline uint32
MOD( uint32 a, uint32 b )
{
	return a % b;
}

/*template<>
inline int64
MOD( int64 a, int64 b )
{
	if( a < 0 ) {
		return a % b + b;
	}
	return a % b;
}*/

template< typename NType >
inline NType
Max( NType a, NType b ) {
	if( a<b ) return b;

	return a;
}

template< typename NType >
inline NType
Max( NType a, NType b, NType c ) {
	if( a<b ) return Max( b, c );

	return Max( a, c );
}

template< typename NType >
inline NType
Min( NType a, NType b ) {
	if( a>b ) return b;

	return a;
}

template< typename NType >
inline NType
Min( NType a, NType b, NType c ) {
	if( a>b ) return Min( b, c );

	return Min( a, c );
}

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

/**
 * This class enables converting many arguments to a string.
 **/
class ToString
{
public:
  inline operator std::string() const
  {
       return toString();
  }

  inline std::string toString() const
  {
       return m_ostream.str();
  }

  /** Add argument of any type to the stream. */
  template <class ArgType>
  ToString& operator<<(ArgType const& arg)
  {
       m_ostream << arg;
       return *this;
  }

private:
  std::ostringstream m_ostream;
};


/**
Macro for a better usage of the class defined above.
*/
#define TO_STRING(MSG) ( std::string(ToString() << MSG) )

using namespace M4D::ErrorHandling;

/** @} */

#endif
