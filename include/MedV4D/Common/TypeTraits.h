#ifndef TYPE_TRAITS_H
#define TYPE_TRAITS_H

#include "MedV4D/Common/Types.h"


template< typename Type >
struct TypeTraits;

/**
 * Function used in conversions of integer values. 
 * @param size Size of examined type.
 * @param sign Wheather examined type is signed.
 * @return ID of type with given characteristics if exists, otherwise
 * NTID_UNKNOWN.
 **/
int16
GetNTIDFromSizeAndSign( uint16 size, bool sign );

uint32
GetByteCountFromNTID( int16 ntid );


template< typename T >
struct TypeTraits
{
	typedef T			Type;
	static const int16	NTID = NTID_UNKNOWN;

	//static const bool	Signed = false;
	static const uint16	BitCount = sizeof( Type )*8;


	/*static const Type	Max = true;
	static const Type	Min = false;
	static const Type	Zero = false;
	static const Type	One = true;

	static const Type	CentralValue = false;

	typedef int8		SignedClosestType;
	typedef int8		SuperiorType;
	typedef int8		SuperiorSignedType;
	typedef float32		SuperiorFloatType;*/
	
	static std::string
	Typename()
	{
		return "Noname assigned";
	}
};

template<>
struct TypeTraits< bool >
{
	typedef bool		Type;
	static const int16	NTID = NTID_BOOL;

	static const bool	Signed = false;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = true;
	static const Type	Min = false;
	static const Type	Zero = false;
	static const Type	One = true;

	static const Type	CentralValue = false;

	typedef int8		SignedClosestType;
	typedef int8		SuperiorType;
	typedef int8		SuperiorSignedType;
	typedef float32		SuperiorFloatType;

	static std::string
	Typename()
	{
		return "bool";
	}
};

template<>
struct TypeTraits< int8 >
{
	typedef int8		Type;
	static const int16	NTID = NTID_INT_8;

	static const bool	Signed = true;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = (Type)MAX_INT8;
	static const Type	Min = (Type)(-MAX_INT8-1);
	static const Type	Zero = 0;
	static const Type	One = 1;

	static const Type	CentralValue = 0;

	typedef int8		SignedClosestType;
	typedef int16		SuperiorType;
	typedef int16		SuperiorSignedType;
	typedef float32		SuperiorFloatType;

	static std::string
	Typename()
	{
		return "int8";
	}
};

template<>
struct TypeTraits< uint8 >
{
	typedef uint8		Type;
	static const int16	NTID = NTID_UINT_8;
	
	static const bool	Signed = false;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = (Type)~((Type)0);
	static const Type	Min = (Type)(0);
	static const Type	Zero = 0;
	static const Type	One = 1;

	static const Type	CentralValue = Max / 2;

	typedef int8		SignedClosestType;
	typedef uint16		SuperiorType;
	typedef int16		SuperiorSignedType;
	typedef float32		SuperiorFloatType;

	static std::string
	Typename()
	{
		return "uint8";
	}
};

template<>
struct TypeTraits< int16 >
{
	typedef int16		Type;
	static const int16	NTID = NTID_INT_16;

	static const bool	Signed = true;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = (Type)MAX_INT16;
	static const Type	Min = (Type)(-MAX_INT16-1);
	static const Type	Zero = 0;
	static const Type	One = 1;

	static const Type	CentralValue = 0;

	typedef int16		SignedClosestType;
	typedef int32		SuperiorType;
	typedef int32		SuperiorSignedType;
	typedef float32		SuperiorFloatType;

	static std::string
	Typename()
	{
		return "int16";
	}
};

template<>
struct TypeTraits< uint16 >
{
	typedef uint16		Type;
	static const int16	NTID = NTID_UINT_16;

	static const bool	Signed = false;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = (Type)~((Type)0);
	static const Type	Min = (Type)(0);
	static const Type	Zero = 0;
	static const Type	One = 1;

	static const Type	CentralValue = Max / 2;

	typedef int16		SignedClosestType;
	typedef uint32		SuperiorType;
	typedef int32		SuperiorSignedType;
	typedef float32		SuperiorFloatType;

	static std::string
	Typename()
	{
		return "uint16";
	}
};

template<>
struct TypeTraits< int32 >
{
	typedef int32		Type;
	static const int16	NTID = NTID_INT_32;

	static const bool	Signed = true;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = (Type)MAX_INT32;
	static const Type	Min = (Type)(-MAX_INT32-1);
	static const Type	Zero = 0;
	static const Type	One = 1;

	static const Type	CentralValue = 0;

	typedef int32		SignedClosestType;
	typedef int64		SuperiorType;
	typedef int64		SuperiorSignedType;
	typedef float32		SuperiorFloatType;

	static std::string
	Typename()
	{
		return "int32";
	}
};

template<>
struct TypeTraits< uint32 >
{
	typedef uint32		Type;
	static const int16	NTID = NTID_UINT_32;

	static const bool	Signed = false;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = (Type)~((Type)0);
	static const Type	Min = (Type)(0);
	static const Type	Zero = 0;
	static const Type	One = 1;

	static const Type	CentralValue = Max / 2;

	typedef int32		SignedClosestType;
	typedef uint64		SuperiorType;
	typedef int64		SuperiorSignedType;
	typedef float32		SuperiorFloatType;

	static std::string
	Typename()
	{
		return "uint32";
	}
};

template<>
struct TypeTraits< int64 >
{
	typedef int64		Type;
	static const int16	NTID = NTID_INT_64;

	static const bool	Signed = true;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = (Type)MAX_INT64;
	static const Type	Min = (Type)(-MAX_INT64-1);
	static const Type	Zero = 0;
	static const Type	One = 1;

	static const Type	CentralValue = 0;

	typedef int64		SignedClosestType;
	typedef int64		SuperiorType;
	typedef float32		SuperiorSignedType;
	typedef float32		SuperiorFloatType;

	static std::string
	Typename()
	{
		return "int64";
	}
};

template<>
struct TypeTraits< uint64 >
{
	typedef uint64		Type;
	static const int16	NTID = NTID_UINT_64;

	static const bool	Signed = false;
	static const uint16	BitCount = sizeof( Type )*8;
	static const Type	Max = (Type)~((Type)0);
	static const Type	Min = (Type)(0);
	static const Type	Zero = 0;
	static const Type	One = 1;

	static const Type	CentralValue = Max / 2;

	typedef int64		SignedClosestType;
	typedef uint64		SuperiorType;
	typedef float32		SuperiorSignedType;
	typedef float32		SuperiorFloatType;

	static std::string
	Typename()
	{
		return "uint64";
	}
};

template<>
struct TypeTraits< float32 >
{
	typedef float32		Type;
	static const int16	NTID = NTID_FLOAT_32;

	static const bool	Signed = true;
	static const uint16	BitCount = sizeof( Type )*8;
	static Type		Max;
	static Type		Min;
	static Type		Zero;
	static Type		One;
	static Type		CentralValue;

	typedef float32		SignedClosestType;
	typedef float64		SuperiorType;
	typedef float64		SuperiorSignedType;
	typedef float64		SuperiorFloatType;

	static std::string
	Typename()
	{
		return "float32";
	}
};

template<>
struct TypeTraits< float64 >
{
	typedef float64		Type;
	static const int16	NTID = NTID_FLOAT_64;

	static const bool	Signed = true;
	static const uint16	BitCount = sizeof( Type )*8;
	static Type		Max;
	static Type		Min;
	static Type		Zero;
	static Type		One;
	static Type		CentralValue;

	typedef float64		SignedClosestType;
	typedef float64		SuperiorType;
	typedef float64		SuperiorSignedType;
	typedef float64		SuperiorFloatType;

	static std::string
	Typename()
	{
		return "float64";
	}
};

template< typename NumericType >
int16 GetNumericTypeID()
{ return TypeTraits< NumericType >::NTID; }

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


#endif /*TYPE_TRAITS_H*/

