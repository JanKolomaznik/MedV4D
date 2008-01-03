/**
 *  Supporting class with static members that solve endianess
 *  Everything will be handled in little endian. On big endian
 *  systems appropriate conversion needs to be done. 
 */

#ifndef BYTE_CONVERTER_H
#define BYTE_CONVERTER_H

namespace M4DSupport
{

	class ByteConverter
	{
	public:
		// serializing methods
		static void ToBytes( int16 what, int8 result[]);
		static void ToBytes( int32 what, int8 result[]);
		static void ToBytes( int64 what, int8 result[]);
		static void ToBytes( uint16 what, int8 result[]);
		static void ToBytes( uint32 what, int8 result[]);
		static void ToBytes( uint64 what, int8 result[]);
		static void ToBytes( float32 what, int8 result[]);
		static void ToBytes( float64 what, int8 result[]);

		// reserializings
		static int16 ToInt16( int8 what[]);
		static int32 ToInt32( int8 what[]);
		static int64 ToInt64( int8 what[]);
		static uint16 ToUInt16( int8 what[]);
		static uint32 ToUInt32( int8 what[]);
		static uint64 ToUInt64( int8 what[]);
		static float32 ToFloat32( int8 what[]);
		static float64 ToFloat64( int8 what[]);

	private:
		  union
		  {
			float32 f;
			unsigned char b[4];
		  } dat1, dat2;

		  union
		  {
			float64 f;
			unsigned char b[8];
		  } dat21, dat22;
	};

}

#endif