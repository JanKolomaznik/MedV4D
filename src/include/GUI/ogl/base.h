#ifdef _MSC_VER
#  pragma once
#endif

#ifndef _BASE_H
#define _BASE_H

/**
 * Basic math, technical stuff, etc.
 * License: General Public License GPL (http://www.gnu.org/copyleft/gpl.html)
 *
 * @author  Josef Pelikan $Author: pepca $
 * @version $Rev: 81 $
 * @date    $Date: 2008-04-08 21:18:24 +0200 (Tue, 08 Apr 2008) $
 */

#include "GUI/ogl/config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
#include <stdarg.h>

//--- macros, types --------------------------------------

#define FLOAT(x)    (static_cast<Float>(x))
#define TYPE(x)     (static_cast<Type>(x))
#define TYPE_PTR(x) (static_cast<Type*>(x))

#define SINGLE(x)   (static_cast<float>(x))
#define DOUBLE(x)   (static_cast<double>(x))
#define INT(x)      (static_cast<int>(x))

#define GLFLOAT(x)  (static_cast<GLfloat>(x))
#define GLINT(x)    (static_cast<GLint>(x))

typedef signed   char  int8;
typedef unsigned char  unsigned8;
typedef          short int16;
typedef unsigned short unsigned16;
typedef          int   int32;
typedef unsigned int   unsigned32;

#ifdef _WIN32

typedef          __int64 int64;
typedef unsigned __int64 unsigned64;

#else

typedef          long long int64;
typedef unsigned long long unsigned64;

#endif

const int8       MAX_I8  = 0x7f;
const int8       MIN_I8  = -MAX_I8-1;
const unsigned8  MAX_U8  = 0xff;
const int16      MAX_I16 = 0x7fff;
const int16      MIN_I16 = -MAX_I16-1;
const unsigned16 MAX_U16 = 0xffff;
const int32      MAX_I32 = 0x7fffffff;
const int32      MIN_I32 = -MAX_I32-1;
const unsigned32 MAX_U32 = 0xffffffff;
#if defined(_MSC_VER) && _MSC_VER < 1300
const int64      MAX_I64 = 0x7fffffffffffffff;
const unsigned64 MAX_U64 = 0xffffffffffffffff;
#else
const int64      MAX_I64 = 0x7fffffffffffffffLL;
const unsigned64 MAX_U64 = 0xffffffffffffffffULL;
#endif
const int64      MIN_I64 = -MAX_I64-1;

#ifndef Assert
#  define Assert(x)      assert(x)
#endif

//--- arithmetic, data -----------------------------------

#define Min(x,y)          ((x)<(y)?(x):(y))
#define Max(x,y)          ((x)>(y)?(x):(y))
#define Zero(x)           memset((void*)&x,0,sizeof(x))
#define Floor(x,min)      {if((x)<(min))(x)=(min);}
#define Ceiling(x,max)    {if((x)>(max))(x)=(max);}
#define Clamp(x,min,max)  {if((x)<(min))(x)=(min);else if((x)>(max))(x)=(max);}
#define Periodic(x,per)   {if((x)<0)(x)=(per)-1+(((x)+1)%(per));else if((x)>=(per))x%=(per);}
#define Del(x)            {if(x){delete x;x=NULL;}}
#define DelArr(x)         {if(x){delete[]x;x=NULL;}}
#define VectorSize(x,y,z) sqrt((x)*(x)+(y)*(y)+(z)*(z))

/// Returns heap-allocated copy of an array.
void *memCopy ( const void *mem, int size );

/// Returns true if file can be opened for reading.
bool fexist ( const char *fileName );

/// Returns prime number at least of size "s".
int primeSize ( int s );

/// 32-bit hash value that should be taken modulo prime.
unsigned hash32 ( const char* str );

#ifdef _MSC_VER

#  define strcasecmp(a,b) _stricmp(a,b)

#else

//#  define strcasecmp(a,b) strcasecmp(a,b)

#endif

//--- color conversions ----------------------------------

extern unsigned grayWeights[4];             // gray weight coefficients for RGB
const int GRAY_SHIFT = 15;                  // sum of grayWeights[] is (1 << GrayShift)

/// RGB -> Y conversion.
#define RGB_2_GRAY(R,G,B) (((R)*grayWeights[0]+(G)*grayWeights[1]+(B)*grayWeights[2])>>GRAY_SHIFT)

/// RGB -> HSV conversion.
template < typename Float>
void rgb2hsv ( int R, int G, int B, Float &H, Float &S, Float &V );

//--- pointers -------------------------------------------

/**
 * Referencable object base class.
 * Implements reference counter.
 */
class RefTarget
{

protected:

  MUTABLE int refCounter;

  inline RefTarget ()
  {
    refCounter = 0;
  }

public:

  inline void addReference () CONST
  {
    refCounter++;
  }

  inline void removeReference ()
  {
    if ( --refCounter <= 0 ) delete this;
  }

  inline int getReference () const
  {
    return refCounter;
  }

  virtual ~RefTarget ()
  {}

};

/**
 * Reference type.
 * Replacement for "Type *".
 * Regular pointers to RefTarget-based objects should be avoided.
 */
template < typename Type >
class Ref
{

protected:

  Type *p;

  inline void reference ( Type *ptr )
  {
    p = ptr;
    if ( ptr )
      p->addReference();
  }

  inline void dereference () const
  {
    if ( p )
      p->removeReference();
  }

public:

  /// Public accessible target type.
  typedef Type TargetType;

  //--- construction ---------------------------------------

  inline Ref ()
  {
    p = NULL;
  }

  inline Ref ( Type *ptr )
  {
    reference(ptr);
  }

  template < typename Desc >
  inline Ref ( const Ref<Desc> &r )
  {
    reference(r.p);
  }

  inline Ref ( const Ref &r )
  {
    reference(r.p);
  }

  inline ~Ref ()
  {
    dereference();
  }

  //--- operators ------------------------------------------

  inline Ref& operator = ( Type *ptr )
  {
    if ( ptr != p )
    {
      dereference();
      reference(ptr);
    }
    return *this;
  }

  inline Ref& operator = ( const Ref &r )
  {
    if ( r.p != p )
    {
      dereference();
      reference(r.p);
    }
    return *this;
  }

  inline Type* operator -> () const
  {
    return p;
  }

  inline operator Type* () const
  {
    return p;
  }

  inline Type& operator * () const
  {
    return *p;
  }

  inline bool operator == ( const Ref &r ) const
  {
    return( p == r.p );
  }

  inline bool operator != ( const Ref &r ) const
  {
    return( p != r.p );
  }

  inline bool isNull () const
  {
    return( p == NULL );
  }

  inline bool notNull () const
  {
    return( p != NULL );
  }

  inline operator bool () const
  {
    return( p != NULL );
  }

  inline bool operator ! () const
  {
    return( p == NULL );
  }

};

//--- system time ----------------------------------------

/// System startup time.
extern unsigned64 startTime;

/**
 * Initialize the system timer (for high-performance timers on Win32).
 */
void startSystemTime ();

/**
 * Actual system time.
 * @return Actual system time in micro-seconds.
 */
unsigned64 getSystemTime ();

/**
 * Application time.
 * @return Actual system time in seconds.
 */
double getAppTime ();

/**
 * Get system clock (high-performacne real-time clock) frequency in Hz.
 */
unsigned getClockFrequency ();

//--- sleep API ------------------------------------------

#ifdef _WIN32

/// Sleep in milliseconds.
#  define SLEEP_MS(t) Sleep(t)

#else

/// Sleep in milliseconds.
#  define SLEEP_MS(t) sleepMs(t)

extern void sleepMs ( unsigned ms );

#endif

//--- floating point math --------------------------------

/**
 * Math functions, constants, etc.
 */
template < typename Float >
class Math
{

public:

    // constants:

  inline static Float eps ();

  inline static Float epsSquare ()
  {
    return eps() * eps();
  }

  inline static Float pi ()
  {
    return FLOAT( 3.141592653589793 );
  }

  inline static Float e ()
  {
    return FLOAT( 2.718281828459045 );
  }

    // functions:

  inline static bool isZero ( Float x )
  {
    return( x > -eps() && x < eps() );
  }

  inline static bool isNegative ( Float x )
  {
    return( x < FLOAT(0) );
  }

  inline static bool isPositive ( Float x )
  {
    return( x > FLOAT(0) );
  }

  inline static int floor ( Float x )
  {
    return INT( ::floor(x) );
  }

  inline static int ceil ( Float x )
  {
    return INT( ::ceil(x) );
  }

  inline static int round ( Float x )
  {
    return floor( x + FLOAT(0.5) );
  }

  inline static Float degreesToRadians ( Float a )
  {
    return( a * (pi()/FLOAT(180)) );
  }

  inline static Float radiansToDegrees ( Float a )
  {
    return( a * (FLOAT(180)/pi()) );
  }

  inline static Float random ()
  {
    return( ::rand() / FLOAT(RAND_MAX) );
  }

  inline static void randomize ( unsigned seed )
  {
    ::srand( seed );
  }

  inline static double distance3 ( const Float *v )
  {
    return sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] );
  }

  inline static double distance4 ( const Float *v )
  {
    return sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3] );
  }

  inline static Float normalize3 ( Float *v )
  {
    double s = distance3(v);
    if ( Math<double>::isZero(s) ) return FLOAT(0);
    double k = 1.0 / s;
    v[0] = FLOAT( k * v[0] );
    v[1] = FLOAT( k * v[1] );
    v[2] = FLOAT( k * v[2] );
    return FLOAT(s);
  }

  inline static Float normalize4 ( Float *v )
  {
    double s = distance4(v);
    if ( Math<double>::isZero(s) ) return FLOAT(0);
    double k = 1.0 / s;
    v[0] = FLOAT( k * v[0] );
    v[1] = FLOAT( k * v[1] );
    v[2] = FLOAT( k * v[2] );
    v[3] = FLOAT( k * v[3] );
    return FLOAT(s);
  }

  static void matrixMultiply3x3 ( const Float *left, const Float *right, Float *result );

  static void vector3TimesMatrix3x3 ( const Float *v, const Float *m, Float *result );

  static void matrix3x3TimesVector3 ( const Float *m, const Float *v, Float *result );

  static void matrixMultiply4x4 ( const Float *left, const Float *right, Float *result );

  static void vector4TimesMatrix4x4 ( const Float *v, const Float *m, Float *result );

  static void vector3TimesMatrix4x4 ( const Float *v, const Float *m, Float *result );

  static void matrix4x4TimesVector4 ( const Float *m, const Float *v, Float *result );

  static void matrix4x4TimesVector3 ( const Float *m, const Float *v, Float *result );

  static void matrixTranspose3x3 ( Float *m );

  static void matrixTranspose4x4 ( Float *m );

  static void quaternionMultiply ( const Float *q, const Float *r, Float *result );

  static void unitQuaternion ( const Float *axis, Float angle, Float *result );

  static void quaternionRotate3 ( const Float *p, const Float *q, Float *result );

  static void quaternionRotate4 ( const Float *p, const Float *q, Float *result );

  static void quaternionToMatrix3x3 ( const Float *q, Float *result );

  static void quaternionToMatrix4x4 ( const Float *q, Float *result );

  static void matrix3x3ToQuaternion ( const Float *m, Float *result );

  static void quaternionSlerp ( const Float *q, const Float *r, Float t, Float *result );

};

template <>
inline float Math<float>::eps ()
{
  return 5.0e-4f;
}

template <>
inline double Math<double>::eps ()
{
  return 1.0e-6;
}

#endif
