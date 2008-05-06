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


template< typename NumericType >
int GetNumericTypeID()
{ return NTID_VOID; }

template<>
int GetNumericTypeID<signed char>()
{ return NTID_SIGNED_CHAR; }

template<>
int GetNumericTypeID<unsigned char>()
{ return NTID_UNSIGNED_CHAR; }

template<>
int GetNumericTypeID<short>()
{ return NTID_SHORT; }

template<>
int GetNumericTypeID<unsigned short>()
{ return NTID_UNSIGNED_SHORT; }

template<>
int GetNumericTypeID<int>()
{ return NTID_INT; }

template<>
int GetNumericTypeID<unsigned int>()
{ return NTID_UNSIGNED_INT; }

template<>
int GetNumericTypeID<long>()
{ return NTID_LONG; }

template<>
int GetNumericTypeID<unsigned long>()
{ return NTID_UNSIGNED_LONG; }

template<>
int GetNumericTypeID<long long>()
{ return NTID_LONG_LONG; }

template<>
int GetNumericTypeID<unsigned long long>()
{ return NTID_UNSIGNED_LONG_LONG; }

template<>
int GetNumericTypeID<float>()
{ return NTID_FLOAT; }

template<>
int GetNumericTypeID<double>()
{ return NTID_DOUBLE; }

template<>
int GetNumericTypeID<bool>()
{ return NTID_BOOL; }


#endif
