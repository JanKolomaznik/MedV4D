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

/**
 * typedef of basic data types for simplier porting
 */
// signed
typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long int64;
// unsigned
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;
// floats
typedef float float32;
typedef double float64;
// others
//typedef uint32 size_t;

enum NumericTypeIDs{ 
	NTID_UNKNOWN,
	NTID_VOID, 

	NTID_INT_8,
	NTID_UINT_8,

	NTID_INT_16,
	NTID_UINT_16,

	NTID_INT_32,
	NTID_UINT_32,

	NTID_INT_64,
	NTID_UINT_64,

	NTID_FLOAT_32,
	NTID_FLOAT_64,

	NTID_BOOL
};

//*****************************************************************************
/**
 * \defgroup TYPE_DEFINE_MACROS Macros defining type belonging 
 * to numeric type ID.
 **/
#define NTID_VOID_TYPE_DEFINE_MACRO			void
#define NTID_INT_8_TYPE_DEFINE_MACRO			int8
#define NTID_UINT_8_TYPE_DEFINE_MACRO			uint8
#define NTID_INT_16_TYPE_DEFINE_MACRO			int16
#define NTID_UINT_16_TYPE_DEFINE_MACRO			uint16
#define NTID_INT_32_TYPE_DEFINE_MACRO			int32
#define NTID_UINT_32_TYPE_DEFINE_MACRO			uint32
#define NTID_INT_64_TYPE_DEFINE_MACRO			int64
#define NTID_UINT_64_TYPE_DEFINE_MACRO			uint64
#define NTID_FLOAT_32_TYPE_DEFINE_MACRO			float32
#define NTID_FLOAT_64_TYPE_DEFINE_MACRO			float64
#define NTID_BOOL_TYPE_DEFINE_MACRO			bool
//*****************************************************************************
#define TYPE_FROM_ID_MACRO( TYPE_ID )\
	TYPE_ID##_TYPE_DEFINE_MACRO

//*****************************************************************************
/**
 * Macro used in NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO() macro. DO NOT USE DIRECTLY!!!
 * @param ID ID of numeric type.
 **/
#define NUMERIC_TYPE_TEMPLATE_CASE_MACRO( ID, ... )\
	case ID: { typedef TYPE_FROM_ID_MACRO( ID ) TTYPE;\
			__VA_ARGS__; };\
		break;
//*****************************************************************************
/**
 * Macro for easy generic programing - mixing static and dynamic polymorhism.
 * Create switch command with cases belonging to numerical types without bool and void.
 * in case templated command is called with apropriete type.
 * @param SWITCH Statement which will be placed in "switch( SWITCH )".
 * @param DEFAULT Code, which will be put in default branch.
 * @param ... Templated command. Must contain TTYPE, which will be replaced by 
 * apropriete type. Example : function_name< TTYPE >( TTYPE* arg )
 **/
#define NUMERIC_TYPE_TEMPLATE_SWITCH_DEFAULT_MACRO( SWITCH, DEFAULT, ... ) \
	switch( SWITCH ) {\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_INT_8, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_8, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_INT_16, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_16, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_INT_32, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_32, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_INT_64, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_64, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_FLOAT_32, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_FLOAT_64, __VA_ARGS__ )\
	default: DEFAULT;\
	}

/**
 * Like previous, but with assert in default branch.
 **/
#define NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( SWITCH, ... ) \
	NUMERIC_TYPE_TEMPLATE_SWITCH_DEFAULT_MACRO( SWITCH, ASSERT( false ), __VA_ARGS__ )

/**
 * Same as previous, but only for integer types.
 * @param SWITCH Statement which will be placed in "switch( SWITCH )".
 * @param ... Templated command. Must contain TTYPE, which will be replaced by 
 * apropriete type. Example : function_name< TTYPE >( TTYPE* arg )
 **/
#define INTEGER_TYPE_TEMPLATE_SWITCH_MACRO( SWITCH, ... ) \
	switch( SWITCH ) {\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_INT_8, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_8, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_INT_16, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_16, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_INT_32, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_32, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_INT_64, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_64, __VA_ARGS__ )\
	default: ASSERT( false );\
	}
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

template< typename NumericType >
int16 GetNumericTypeID()
{ return NTID_UNKNOWN; }

template<>
int16 GetNumericTypeID<int8>();

template<>
int16 GetNumericTypeID<uint8>();

template<>
int16 GetNumericTypeID<int16>();

template<>
int16 GetNumericTypeID<uint16>();

template<>
int16 GetNumericTypeID<int32>();

template<>
int16 GetNumericTypeID<uint32>();

template<>
int16 GetNumericTypeID<int64>();

template<>
int16 GetNumericTypeID<uint64>();

template<>
int16 GetNumericTypeID<float32>();

template<>
int16 GetNumericTypeID<float64>();

template<>
int16 GetNumericTypeID<bool>();


//TODO - platform independend.
/**
 * Function used in conversions of integer values. 
 * @param size Size of examined type.
 * @param sign Wheather examined type is signed.
 * @return ID of type with given characteristics if exists, otherwise
 * NTID_UNKNOWN.
 **/
int16
GetNTIDFromSizeAndSign( uint16 size, bool sign );


//TODO - move
#define PROHIBIT_COPYING_OF_OBJECT_MACRO( ClassName ) \
	ClassName( const ClassName& ); \
	ClassName& \
	operator=( const ClassName& ); 


// these are used in every file so include them in one place
// here because Common are used in every file as well
#include "Debug.h"
#include "Log.h"
#include "ExceptionBase.h"
#include <iomanip>

//TODO test and move to better place
#define MAKESTRING( S ) #S

using namespace M4D::ErrorHandling;

#endif
