/**
 *  @ingroup common
 *  @file Common.cpp
 *  @author Jan Kolomaznik
 */
#include "Common.h"
#include "Types.h"

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

