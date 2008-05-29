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
#define NUMERIC_TYPE_TEMPLATE_CASE_MACRO( ID, TEMPLATE_EXP )\
	case ID: { typedef TYPE_FROM_ID_MACRO( ID ) TTYPE;\
			TEMPLATE_EXP; };\
		break;
//*****************************************************************************
/**
 * Macro for easy generic programing - mixing static and dynamic polymorhism.
 * Create switch command with cases belonging to numerical types without bool and void.
 * in case templated command is called with apropriete type.
 * @param SWITCH Statement which will be placed in "switch( SWITCH )".
 * @param TEMPLATE_EXP Templated command. Must contain TTYPE, which will be replaced by 
 * apropriete type. Example : function_name< TTYPE >( TTYPE* arg )
 **/
#define NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( SWITCH, TEMPLATE_EXP ) \
	switch( SWITCH ) {\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_SIGNED_CHAR, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_CHAR, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_SHORT, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_SHORT, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_INT, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_INT, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_LONG, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_LONG, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_LONG_LONG, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_LONG_LONG, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_FLOAT, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_DOUBLE, TEMPLATE_EXP )\
	default: ASSERT( false );\
	}
/**
 * Same as previous, but only for integer types.
 * @param SWITCH Statement which will be placed in "switch( SWITCH )".
 * @param TEMPLATE_EXP Templated command. Must contain TTYPE, which will be replaced by 
 * apropriete type. Example : function_name< TTYPE >( TTYPE* arg )
 **/
#define INTEGER_TYPE_TEMPLATE_SWITCH_MACRO( SWITCH, TEMPLATE_EXP ) \
	switch( SWITCH ) {\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_SIGNED_CHAR, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_CHAR, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_SHORT, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_SHORT, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_INT, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_INT, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_LONG, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_LONG, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_LONG_LONG, TEMPLATE_EXP )\
	NUMERIC_TYPE_TEMPLATE_CASE_MACRO( NTID_UNSIGNED_LONG_LONG, TEMPLATE_EXP )\
	default: ASSERT( false );\
	}
//*****************************************************************************

template< typename NumericType >
int GetNumericTypeID()
{ return NTID_VOID; }

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
 * NTID_VOID.
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
