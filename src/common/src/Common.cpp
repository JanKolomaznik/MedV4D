/**
 *  @ingroup common
 *  @file Common.cpp
 *  @author Jan Kolomaznik
 */
#include "common/Common.h"


const float32 Epsilon = 1.0E-10;
const float32 PI = 3.141592f;
const float32 PIx2 = PI*2;
const float32 PIx3 = PI*3;
const float32 PIx4 = PI*4;
const float32 PId2 = PI/2;
const float32 PId3 = PI/3;
const float32 PId4 = PI/4;


#include "common/Direction.h"

Vector<int32,2>	directionOffset[] = {
	Vector<int32,2>( 1, 0 ),
	Vector<int32,2>( 1, -1 ),
	Vector<int32,2>( 0, -1 ),
	Vector<int32,2>( -1, -1 ),
	Vector<int32,2>( -1, 0 ),
	Vector<int32,2>( -1, 1 ),
	Vector<int32,2>( 0, 1 ),
	Vector<int32,2>( 1, 1 )
	};


#include "common/Types.h"

float32 TypeTraits< float32 >::Max = MAX_FLOAT32;
float32 TypeTraits< float32 >::Min = -MAX_FLOAT32;
float32 TypeTraits< float32 >::Zero = 0.0f;
float32 TypeTraits< float32 >::One = 1.0f;
float32 TypeTraits< float32 >::CentralValue = 0.0f;

float64 TypeTraits< float64 >::Max = MAX_FLOAT64;
float64 TypeTraits< float64 >::Min = -MAX_FLOAT64;
float64 TypeTraits< float64 >::Zero = 0.0;
float64 TypeTraits< float64 >::One = 1.0;
float64 TypeTraits< float64 >::CentralValue = 0.0;

#define INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( TTYPE, DIM ) \
		template<> SimpleVector< TTYPE, DIM > TypeTraits< SimpleVector< TTYPE, DIM > >::Max = SimpleVector< TTYPE, DIM >( TypeTraits< TTYPE >::Max ); \
		template<> SimpleVector< TTYPE, DIM > TypeTraits< SimpleVector< TTYPE, DIM > >::Min = SimpleVector< TTYPE, DIM >( TypeTraits< TTYPE >::Min ); \
		template<> SimpleVector< TTYPE, DIM > TypeTraits< SimpleVector< TTYPE, DIM > >::Zero = SimpleVector< TTYPE, DIM >( TypeTraits< TTYPE >::Zero ); \
		template<> SimpleVector< TTYPE, DIM > TypeTraits< SimpleVector< TTYPE, DIM > >::CentralValue = SimpleVector< TTYPE, DIM >( TypeTraits< TTYPE >::CentralValue ); \
		template<> TTYPE TypeTraits< SimpleVector< TTYPE, DIM > >::One = TypeTraits< TTYPE >::One;


INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( int8, 2 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( uint8, 2 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( int16, 2 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( uint16, 2 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( int32, 2 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( uint32, 2 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( int64, 2 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( uint64, 2 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( float32, 2 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( float64, 2 )

INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( int8, 3 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( uint8, 3 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( int16, 3 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( uint16, 3 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( int32, 3 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( uint32, 3 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( int64, 3 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( uint64, 3 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( float32, 3 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( float64, 3 )

INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( int8, 4 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( uint8, 4 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( int16, 4 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( uint16, 4 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( int32, 4 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( uint32, 4 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( int64, 4 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( uint64, 4 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( float32, 4 )
INITIALIZE_SIMPLE_VECTOR_CONSTANTS_MACRO( float64, 4 )

//TODO - platform independend.
int16
GetNTIDFromSizeAndSign( uint16 size, bool sign )
{
	switch ( size ) {
	case 1:
		return ( sign ? NTID_INT_8 : NTID_UINT_8 );
	case 2:
		return ( sign ? NTID_INT_16 : NTID_UINT_16 );
	case 4:
		return ( sign ? NTID_INT_32 : NTID_UINT_32 );
	case 8:
		return ( sign ? NTID_INT_64 : NTID_UINT_64 );
	default:
		return NTID_UNKNOWN;
	}
}

/*
template<>
int16 GetNumericTypeID<int8>()
{ return NTID_INT_8; }

template<>
int16 GetNumericTypeID<uint8>()
{ return NTID_UINT_8; }

template<>
int16 GetNumericTypeID<int16>()
{ return NTID_INT_16; }

template<>
int16 GetNumericTypeID<uint16>()
{ return NTID_UINT_16; }

template<>
int16 GetNumericTypeID<int32>()
{ return NTID_INT_32; }

template<>
int16 GetNumericTypeID<uint32>()
{ return NTID_UINT_32; }

template<>
int16 GetNumericTypeID<int64>()
{ return NTID_INT_64; }

template<>
int16 GetNumericTypeID<uint64>()
{ return NTID_UINT_64; }

template<>
int16 GetNumericTypeID<float32>()
{ return NTID_FLOAT_32; }

template<>
int16 GetNumericTypeID<float64>()
{ return NTID_FLOAT_64; }

template<>
int16 GetNumericTypeID<bool>()
{ return NTID_BOOL; }
*/

