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


template< typename Type >
struct TypeTraits;

template<>
struct TypeTraits< int8 >
{
	typedef int8		Type;
	
	static const bool	Signed = true;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = (Type)MAX_INT8;
	static const Type	Min = (Type)(-MAX_INT8-1);

	typedef int64		SuperiorType;
};

template<>
struct TypeTraits< uint8 >
{
	typedef uint8		Type;
	
	static const bool	Signed = false;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = ~((Type)0);
	static const Type	Min = (Type)(0);

	typedef uint64		SuperiorType;
};

template<>
struct TypeTraits< int16 >
{
	typedef int16		Type;

	static const bool	Signed = true;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = (Type)MAX_INT16;
	static const Type	Min = (Type)(-MAX_INT16-1);

	typedef int64		SuperiorType;
};

template<>
struct TypeTraits< uint16 >
{
	typedef uint16		Type;

	static const bool	Signed = false;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = ~((Type)0);
	static const Type	Min = (Type)(0);

	typedef uint64		SuperiorType;
};

template<>
struct TypeTraits< int32 >
{
	typedef int32		Type;

	static const bool	Signed = true;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = (Type)MAX_INT32;
	static const Type	Min = (Type)(-MAX_INT32-1);

	typedef int64		SuperiorType;
};

template<>
struct TypeTraits< uint32 >
{
	typedef uint32		Type;

	static const bool	Signed = false;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = ~((Type)0);
	static const Type	Min = (Type)(0);

	typedef uint64		SuperiorType;
};

template<>
struct TypeTraits< int64 >
{
	typedef int64		Type;

	static const bool	Signed = true;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = (Type)MAX_INT64;
	static const Type	Min = (Type)(-MAX_INT64-1);

	typedef int64		SuperiorType;
};

template<>
struct TypeTraits< uint64 >
{
	typedef uint64		Type;

	static const bool	Signed = false;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = ~((Type)0);
	static const Type	Min = (Type)(0);

	typedef uint64		SuperiorType;
};

template<>
struct TypeTraits< float32 >
{
	typedef float32		Type;

	static const bool	Signed = true;
	static const uint16	BitCount = sizeof( Type )*8;
	static Type		Max;
	static Type		Min;

	typedef float64		SuperiorType;
};

template<>
struct TypeTraits< float64 >
{
	typedef float64		Type;

	static const bool	Signed = true;
	static const uint16	BitCount = sizeof( Type )*8;
	static Type		Max;
	static Type		Min;

	typedef float64		SuperiorType;
};

//********************************************************************
template< uint16 size, bool sign >
struct IntegerType;

template<>
struct IntegerType< 1, true >
{
	typedef int8		Type;

	static const bool	Signed = TypeTraits< Type >::Signed;
	static const uint16	BitCount = TypeTraits< Type >::BitCount;
	static const Type	Max = TypeTraits< Type >::Max;
	static const Type	Min = TypeTraits< Type >::Min;
};

template<>
struct IntegerType< 1, false >
{
	typedef uint8	Type;

	static const bool	Signed = TypeTraits< Type >::Signed;
	static const uint16	BitCount = TypeTraits< Type >::BitCount;
	static const Type	Max = TypeTraits< Type >::Max;
	static const Type	Min = TypeTraits< Type >::Min;
};

template<>
struct IntegerType< 2, true >
{
	typedef int16	Type;

	static const bool	Signed = TypeTraits< Type >::Signed;
	static const uint16	BitCount = TypeTraits< Type >::BitCount;
	static const Type	Max = TypeTraits< Type >::Max;
	static const Type	Min = TypeTraits< Type >::Min;
};

template<>
struct IntegerType< 2, false >
{
	typedef uint16	Type;

	static const bool	Signed = TypeTraits< Type >::Signed;
	static const uint16	BitCount = TypeTraits< Type >::BitCount;
	static const Type	Max = TypeTraits< Type >::Max;
	static const Type	Min = TypeTraits< Type >::Min;
};

template<>
struct IntegerType< 4, true >
{
	typedef int32	Type;

	static const bool	Signed = TypeTraits< Type >::Signed;
	static const uint16	BitCount = TypeTraits< Type >::BitCount;
	static const Type	Max = TypeTraits< Type >::Max;
	static const Type	Min = TypeTraits< Type >::Min;
};

template<>
struct IntegerType< 4, false >
{
	typedef uint32	Type;

	static const bool	Signed = TypeTraits< Type >::Signed;
	static const uint16	BitCount = TypeTraits< Type >::BitCount;
	static const Type	Max = TypeTraits< Type >::Max;
	static const Type	Min = TypeTraits< Type >::Min;
};

template<>
struct IntegerType< 8, true >
{
	typedef int64	Type;

	static const bool	Signed = TypeTraits< Type >::Signed;
	static const uint16	BitCount = TypeTraits< Type >::BitCount;
	static const Type	Max = TypeTraits< Type >::Max;
	static const Type	Min = TypeTraits< Type >::Min;
};

template<>
struct IntegerType< 8, false >
{
	typedef uint64	Type;

	static const bool	Signed = TypeTraits< Type >::Signed;
	static const uint16	BitCount = TypeTraits< Type >::BitCount;
	static const Type	Max = TypeTraits< Type >::Max;
	static const Type	Min = TypeTraits< Type >::Min;
};

/** @} */

#endif/*TYPES_H*/
