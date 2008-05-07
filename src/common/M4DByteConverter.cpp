
#include "Common.h"
#include "M4DByteConverter.h"

namespace M4DSupport
{

	//////////////////////////
	// serializings
	//////////////////////////
	void
	ByteConverter::ToBytes( int16 what, int8 result[])
	{
		result[0] = what & 255;
		result[1] = ( what >> 8 ) & 255;
	}

	void
	ByteConverter::ToBytes( int32 what, int8 result[])
	{
		result[0] = what & 255;
		result[1] = ( what >> 8 ) & 255;
		result[2] = ( what >> 16 ) & 255;
		result[3] = ( what >> 24 ) & 255;
	}

	void
	ByteConverter::ToBytes( int64 what, int8 result[])
	{
		result[0] = (int8) what & 255;
		result[1] = (int8) ( what >> 8 ) & 255;
		result[2] = (int8) ( what >> 16 ) & 255;
		result[3] = (int8) ( what >> 24 ) & 255;
		result[4] = (int8) ( what >> 32 ) & 255;
		result[5] = (int8) ( what >> 40 ) & 255;
		result[6] = (int8) ( what >> 48 ) & 255;
		result[7] = (int8) ( what >> 56 ) & 255;
	}

	void
	ByteConverter::ToBytes( uint16 what, int8 result[])
	{
		// just convert to signed version and pass
		ToBytes( (int16) what, result);
	}

	void
	ByteConverter::ToBytes( uint32 what, int8 result[])
	{
		ToBytes( (int32) what, result);
	}

	void
	ByteConverter::ToBytes( uint64 what, int8 result[])
	{
		ToBytes( (int64) what, result);
	}

	void
	ByteConverter::ToBytes( float32 what, int8 result[])
	{

	}

	void
	ByteConverter::ToBytes( float64 what, int8 result[])
	{
	}
	/////////////////////////////////////////////////////////////
	// reserializings
	int16 
	ByteConverter::ToInt16( int8 what[])
	{
		return *( (int16*) what);
	}

	int32
	ByteConverter::ToInt32( int8 what[])
	{
		return *( (int32*) what);
	}

	int64
	ByteConverter::ToInt64( int8 what[])
	{
		return *( (int64*) what);
	}

	uint16
	ByteConverter::ToUInt16( int8 what[])
	{
		return *( (uint16*) what);
	}

	uint32
	ByteConverter::ToUInt32( int8 what[])
	{
		return *( (uint32*) what);
	}

	uint64
	ByteConverter::ToUInt64( int8 what[])
	{
		return *( (uint64*) what);
	}

	float32
	ByteConverter::ToFloat32( int8 what[])
	{
		return *( (float32*) what);
	}

	float64
	ByteConverter::ToFloat64( int8 what[])
	{
		return *( (float64*) what);
	}

}
