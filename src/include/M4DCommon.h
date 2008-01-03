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
typedef int32 ssize_t;
typedef int8 retval_t;

/**
 * return values constants
 */
#define E_OK			0
#define E_FAILED		-1


#endif
