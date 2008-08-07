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
	NTID_SIGNED_CHAR,
	NTID_UNSIGNED_CHAR,
	NTID_SHORT,
	NTID_UNSIGNED_SHORT,
	NTID_INT,
	NTID_UNSIGNED_INT,
	NTID_LONG,
	NTID_UNSIGNED_LONG,
	NTID_LONG_LONG,
	NTID_UNSIGNED_LONG_LONG,
	NTID_FLOAT,
	NTID_DOUBLE,
	NTID_BOOL
};

//*****************************************************************************
/**
 * \defgroup TYPE_DEFINE_MACROS Macros defining type belonging 
 * to numeric type ID.
 **/
#define NTID_VOID_TYPE_DEFINE_MACRO			void
#define NTID_SIGNED_CHAR_TYPE_DEFINE_MACRO		signed char
#define NTID_UNSIGNED_CHAR_TYPE_DEFINE_MACRO		unsigned char
#define NTID_SHORT_TYPE_DEFINE_MACRO			short
#define NTID_UNSIGNED_SHORT_TYPE_DEFINE_MACRO		unsigned short
#define NTID_INT_TYPE_DEFINE_MACRO			int
#define NTID_UNSIGNED_INT_TYPE_DEFINE_MACRO		unsigned int
#define NTID_LONG_TYPE_DEFINE_MACRO			long
#define NTID_UNSIGNED_LONG_TYPE_DEFINE_MACRO		unsigned long
#define NTID_LONG_LONG_TYPE_DEFINE_MACRO		long long
#define NTID_UNSIGNED_LONG_LONG_TYPE_DEFINE_MACRO	unsigned long long
#define NTID_FLOAT_TYPE_DEFINE_MACRO			float
#define NTID_DOUBLE_TYPE_DEFINE_MACRO			double
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
 * @param ... Templated command. Must contain TTYPE, which will be replaced by 
 * apropriete type. Example : function_name< TTYPE >( TTYPE* arg )
 **/
#define NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( SWITCH, ... ) \
	switch( SWITCH ) {\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_SIGNED_CHAR, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_CHAR, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_SHORT, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_SHORT, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_INT, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_INT, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_LONG, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_LONG, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_LONG_LONG, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_LONG_LONG, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_FLOAT, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_DOUBLE, __VA_ARGS__ )\
	default: ASSERT( false );\
	}
/**
 * Same as previous, but only for integer types.
 * @param SWITCH Statement which will be placed in "switch( SWITCH )".
 * @param ... Templated command. Must contain TTYPE, which will be replaced by 
 * apropriete type. Example : function_name< TTYPE >( TTYPE* arg )
 **/
#define INTEGER_TYPE_TEMPLATE_SWITCH_MACRO( SWITCH, ... ) \
	switch( SWITCH ) {\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_SIGNED_CHAR, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_CHAR, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_SHORT, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_SHORT, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_INT, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_INT, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_LONG, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_LONG, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_LONG_LONG, __VA_ARGS__ )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_LONG_LONG, __VA_ARGS__ )\
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


template< typename NumericType >
int GetNumericTypeID()
{ return NTID_UNKNOWN; }

template<>
int GetNumericTypeID<signed char>();

template<>
int GetNumericTypeID<unsigned char>();

template<>
int GetNumericTypeID<short>();

template<>
int GetNumericTypeID<unsigned short>();

template<>
int GetNumericTypeID<int>();

template<>
int GetNumericTypeID<unsigned int>();

template<>
int GetNumericTypeID<long>();

template<>
int GetNumericTypeID<unsigned long>();

template<>
int GetNumericTypeID<long long>();

template<>
int GetNumericTypeID<unsigned long long>();

template<>
int GetNumericTypeID<float>();

template<>
int GetNumericTypeID<double>();

template<>
int GetNumericTypeID<bool>();


//TODO - platform independend.
/**
 * Function used in conversions of integer values. 
 * @param size Size of examined type.
 * @param signed Wheather examined type is signed.
 * @return ID of type with given characteristics if exists, otherwise
 * NTID_UNKNOWN.
 **/
int
GetNTIDFromSizeAndSign( uint8 size, bool sign );


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
