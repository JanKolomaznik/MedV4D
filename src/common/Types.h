#ifndef TYPES_H
#define TYPES_H

/**
 *  @ingroup common
 *  @file Types.h
 *
 *  @addtogroup common
 *  @{
 *  @section types Types definition
 *
 *  Because project is planned to be multi platform. we must use
 *  architecture independent types - we achieved that by typedefs
 *  (int16, uint32, float32 etc. ), which must be rewritten for 
 *  every incompatible platform.
 *
 *  Another issue we are dealing with in project is how to slightly
 *  move from static polymorphism (compile time - templates) and 
 *  dynamic polymorphism (runtime - object programming, RTTI).
 *  For this we have few tools to make that easier.
 *
 *  We have enumeration with values pertaining to basic numeric types. 
 *  These values are returned from templated function 
 *  GetNumericTypeID<TYPE>().
 *  On the other side over these values macros like 
 *  NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO() decide which template instance call. 
 */

#include <ostream>
/*
 * typedef of basic data types for simplier porting
 */
// signed
typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int int64;
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

static const int8 	MAX_INT8 = 0x7F;
static const uint8 	MAX_UINT8 = 0xFF;
static const int16 	MAX_INT16 = 0x7FFF;
static const uint16 	MAX_UINT16 = 0xFFFF;
static const int32 	MAX_INT32 = 0x7FFFFFFF;
static const uint32 	MAX_UINT32 = 0xFFFFFFFF;
static const int64 	MAX_INT64 = 0x7FFFFFFFFFFFFFFFLL;
static const uint64 	MAX_UINT64 = 0xFFFFFFFFFFFFFFFFULL;
static const float32	MAX_FLOAT32 = 1E+37f;
static const float64	MAX_FLOAT64 = 1E+37;

enum NumericTypeIDs{ 
	//Simple numeric types IDs
	NTID_SIMPLE_TYPE_MASK = 0x0F,
	NTID_SIMPLE_TYPES = 0x0,

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

	NTID_BOOL,
	
	NTID_VECTOR_DIM_STEP = 0x10,
	//--------------------------
	NTID_2D_VECTORS = 2 * NTID_VECTOR_DIM_STEP,

	NTID_2D_INT_8,
	NTID_2D_UINT_8,

	NTID_2D_INT_16,
	NTID_2D_UINT_16,

	NTID_2D_INT_32,
	NTID_2D_UINT_32,

	NTID_2D_INT_64,
	NTID_2D_UINT_64,

	NTID_2D_FLOAT_32,
	NTID_2D_FLOAT_64,

	NTID_2D_BOOL,
	//--------------------------
	NTID_3D_VECTORS = 3 * NTID_VECTOR_DIM_STEP,

	NTID_3D_INT_8,
	NTID_3D_UINT_8,

	NTID_3D_INT_16,
	NTID_3D_UINT_16,

	NTID_3D_INT_32,
	NTID_3D_UINT_32,

	NTID_3D_INT_64,
	NTID_3D_UINT_64,

	NTID_3D_FLOAT_32,
	NTID_3D_FLOAT_64,

	NTID_3D_BOOL,
	//--------------------------
	NTID_4D_VECTORS = 4 * NTID_VECTOR_DIM_STEP,

	NTID_4D_INT_8,
	NTID_4D_UINT_8,

	NTID_4D_INT_16,
	NTID_4D_UINT_16,

	NTID_4D_INT_32,
	NTID_4D_UINT_32,

	NTID_4D_INT_64,
	NTID_4D_UINT_64,

	NTID_4D_FLOAT_32,
	NTID_4D_FLOAT_64,

	NTID_4D_BOOL,
	//--------------------------
	//Special types
	/*
	NTID_RGB,
	NTID_RGBA,

	NTID_COMPLEX_INT_8,
	NTID_COMPLEX_INT_16,
	NTID_COMPLEX_INT_32,
	NTID_COMPLEX_INT_64,
	NTID_COMPLEX_FLOAT_32,
	NTID_COMPLEX_FLOAT_64,
	*/
	NTID_UNKNOWN,
	NTID_VOID, 
};

//*****************************************************************************
/**
 * Macros defining type belonging 
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

#define NTID_2D_INT_8_TYPE_DEFINE_MACRO			Vector< int8, 2 >
#define NTID_2D_UINT_8_TYPE_DEFINE_MACRO		Vector< uint8, 2 >
#define NTID_2D_INT_16_TYPE_DEFINE_MACRO		Vector< int16, 2 >
#define NTID_2D_UINT_16_TYPE_DEFINE_MACRO		Vector< uint16, 2 >
#define NTID_2D_INT_32_TYPE_DEFINE_MACRO		Vector< int32, 2 >
#define NTID_2D_UINT_32_TYPE_DEFINE_MACRO		Vector< uint32, 2 >
#define NTID_2D_INT_64_TYPE_DEFINE_MACRO		Vector< int64, 2 >
#define NTID_2D_UINT_64_TYPE_DEFINE_MACRO		Vector< uint64, 2 >
#define NTID_2D_FLOAT_32_TYPE_DEFINE_MACRO		Vector< float32, 2 >
#define NTID_2D_FLOAT_64_TYPE_DEFINE_MACRO		Vector< float64, 2 >
#define NTID_2D_BOOL_TYPE_DEFINE_MACRO			Vector< bool, 2 >

#define NTID_3D_INT_8_TYPE_DEFINE_MACRO			Vector< int8, 3 >
#define NTID_3D_UINT_8_TYPE_DEFINE_MACRO		Vector< uint8, 3 >
#define NTID_3D_INT_16_TYPE_DEFINE_MACRO		Vector< int16, 3 >
#define NTID_3D_UINT_16_TYPE_DEFINE_MACRO		Vector< uint16, 3 >
#define NTID_3D_INT_32_TYPE_DEFINE_MACRO		Vector< int32, 3 >
#define NTID_3D_UINT_32_TYPE_DEFINE_MACRO		Vector< uint32, 3 >
#define NTID_3D_INT_64_TYPE_DEFINE_MACRO		Vector< int64, 3 >
#define NTID_3D_UINT_64_TYPE_DEFINE_MACRO		Vector< uint64, 3 >
#define NTID_3D_FLOAT_32_TYPE_DEFINE_MACRO		Vector< float32, 3 >
#define NTID_3D_FLOAT_64_TYPE_DEFINE_MACRO		Vector< float64, 3 >
#define NTID_3D_BOOL_TYPE_DEFINE_MACRO			Vector< bool, 3 >

#define NTID_4D_INT_8_TYPE_DEFINE_MACRO			Vector< int8, 4 >
#define NTID_4D_UINT_8_TYPE_DEFINE_MACRO		Vector< uint8, 4 >
#define NTID_4D_INT_16_TYPE_DEFINE_MACRO		Vector< int16, 4 >
#define NTID_4D_UINT_16_TYPE_DEFINE_MACRO		Vector< uint16, 4 >
#define NTID_4D_INT_32_TYPE_DEFINE_MACRO		Vector< int32, 4 >
#define NTID_4D_UINT_32_TYPE_DEFINE_MACRO		Vector< uint32, 4 >
#define NTID_4D_INT_64_TYPE_DEFINE_MACRO		Vector< int64, 4 >
#define NTID_4D_UINT_64_TYPE_DEFINE_MACRO		Vector< uint64, 4 >
#define NTID_4D_FLOAT_32_TYPE_DEFINE_MACRO		Vector< float32, 4 >
#define NTID_4D_FLOAT_64_TYPE_DEFINE_MACRO		Vector< float64, 4 >
#define NTID_4D_BOOL_TYPE_DEFINE_MACRO			Vector< bool, 4 >

//*****************************************************************************
#define TYPE_FROM_ID_MACRO( TYPE_ID )\
	TYPE_ID##_TYPE_DEFINE_MACRO

//*****************************************************************************
/**
 * Macro used in NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO() macro. DO NOT USE DIRECTLY!!!
 * @param ID ID of numeric type.
 **/
#define TYPE_TEMPLATE_CASE_MACRO( ID, ... )\
	case ID: { typedef TYPE_FROM_ID_MACRO( ID ) TTYPE;\
			__VA_ARGS__; };\
		break;
//*****************************************************************************
#define TYPE_TEMPLATE_SWITCH_DEFAULT_MACRO( SWITCH, DEFAULT, ... ) \
	switch( SWITCH ) {\
	TYPE_TEMPLATE_CASE_MACRO( NTID_INT_8, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_8, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_INT_16, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_16, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_INT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_INT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_FLOAT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_FLOAT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_2D_INT_8, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_2D_UINT_8, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_2D_INT_16, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_2D_UINT_16, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_2D_INT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_2D_UINT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_2D_INT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_2D_UINT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_2D_FLOAT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_2D_FLOAT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_3D_INT_8, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_3D_UINT_8, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_3D_INT_16, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_3D_UINT_16, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_3D_INT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_3D_UINT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_3D_INT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_3D_UINT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_3D_FLOAT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_3D_FLOAT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_4D_INT_8, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_4D_UINT_8, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_4D_INT_16, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_4D_UINT_16, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_4D_INT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_4D_UINT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_4D_INT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_4D_UINT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_4D_FLOAT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_4D_FLOAT_64, __VA_ARGS__ )\
	default: DEFAULT;\
	}

/**
 * Like previous, but with assert in default branch.
 **/
#define TYPE_TEMPLATE_SWITCH_MACRO( SWITCH, ... ) \
	TYPE_TEMPLATE_SWITCH_DEFAULT_MACRO( SWITCH, ASSERT( false ), __VA_ARGS__ )
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
	TYPE_TEMPLATE_CASE_MACRO( NTID_INT_8, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_8, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_INT_16, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_16, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_INT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_INT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_FLOAT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_FLOAT_64, __VA_ARGS__ )\
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
	TYPE_TEMPLATE_CASE_MACRO( NTID_INT_8, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_8, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_INT_16, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_16, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_INT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_32, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_INT_64, __VA_ARGS__ )\
	TYPE_TEMPLATE_CASE_MACRO( NTID_UINT_64, __VA_ARGS__ )\
	default: ASSERT( false );\
	}


//TODO - platform independend.


/** @} */

#endif/*TYPES_H*/
