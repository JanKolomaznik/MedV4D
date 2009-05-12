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

#include "common/Types.h"
#include "common/TypeComparator.h"
#include "common/Debug.h"
#include "common/Log.h"
#include "common/ExceptionBase.h"
#include "common/Endianess.h"
#include "common/Direction.h"
#include "common/MathTools.h"
#include "common/Vector.h"
#include <iomanip>
#include <sstream>

#define BINSTREAM_WRITE_MACRO( STREAM, VARIABLE ) \
	STREAM.write( (char*)&VARIABLE, sizeof(VARIABLE) );

#define BINSTREAM_READ_MACRO( STREAM, VARIABLE ) \
	STREAM.read( (char*)&VARIABLE, sizeof(VARIABLE) );

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

extern Vector<int32,2>	directionOffset[8];

//***********************************************************

template< unsigned Dim >
Vector< int32, Dim >
StridesFromSize( const Vector< uint32, Dim > &size )
{
	Vector< int32, Dim > result;

	result[0] = 1;
	for( unsigned i = 1; i < Dim; ++i ) {
		result[i] = result[i-1] * size[i-1];
	}
	return result;
}

//***********************************************************

#define SIMPLE_GET_METHOD( TYPE, NAME, PARAM_NAME ) \
	TYPE Get##NAME ()const{ return PARAM_NAME ; }

#define SIMPLE_SET_METHOD( TYPE, NAME, PARAM_NAME ) \
	void Set##NAME ( TYPE value ){ PARAM_NAME = value; }
		
#define SIMPLE_GET_SET_METHODS( TYPE, NAME, PARAM_NAME ) \
	SIMPLE_GET_METHOD( TYPE, NAME, PARAM_NAME ) \
	SIMPLE_SET_METHOD( TYPE, NAME, PARAM_NAME )

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
