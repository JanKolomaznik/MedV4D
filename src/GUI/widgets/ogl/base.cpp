/**
 *  base.cpp - basic implementations.
 *  License: General Public License GPL (http://www.gnu.org/copyleft/gpl.html)
 *
 *  @author  Josef Pelikan $Author: pepca $
 *  @version $Rev: 85 $
 *  @date    $Date: 2008-05-08 08:23:20 +0200 (Thu, 08 May 2008) $
 */

#include "GUI/ogl/base.h"

//--- arithmetic, data -----------------------------------

void *memCopy ( const void *mem, int size )
  // returns heap-allocated copy of an array
{
  if ( !mem ) return NULL;
  void *newArr = new char[size];
  Assert( newArr );
  return memcpy(newArr,mem,size);
}

bool fexist ( const char *fileName )
  // returns true if file can be opened for reading
{
  if ( !fileName || !fileName[0] ) return false;
  FILE *f = fopen(fileName,"r");
  if ( !f ) return false;
  fclose(f);
  return true;
}

const int MAX_PRIME = 20;

int primes[MAX_PRIME] =
{
       13,      31,      61,     127,
      251,     509,    1021,    2039,
     4051,    8111,	  16223,   32467,
    64937,  129887,  259781,  519577,
  1039169, 2078339, 4156709, 8313433
};

int primeSize ( int s )
{
  int i = 0;
  while ( i < MAX_PRIME && primes[i] < s ) i++;
  return primes[i];
}

const int CHAR_BITS = 5;
const int CHAR_MASK = ((1<<CHAR_BITS)-1);

unsigned hash32 ( const char* str )
  // returns 32-bit hash value that should be taken modulo prime
{
  unsigned result = 0xF45A06C1L;            // result for ""
  unsigned buffer = 0x03858377L;            // for short strings

  if ( str )
  {
    int bufbits = 0;
    char ch;
    while ( *str )
    {                                         // next character
      ch = (char)( (*str++) & CHAR_MASK );
      if ( bufbits + CHAR_BITS <= 32 )
      {
        buffer   = (buffer << CHAR_BITS) + ch;
        bufbits += CHAR_BITS;
      }
      else                                    // buffer is full
      {
        result  ^= (buffer << (32 - bufbits)) + (ch >> (bufbits + CHAR_BITS - 32));
        bufbits += CHAR_BITS - 32;
        buffer   = ch & ((1 << bufbits) - 1);
      }
    }
  }
  return( result ^ buffer );
}

//--- color conversions ----------------------------------

unsigned grayWeights[4] =                   // Gray weight coefficients for RGB
{
  9794, 19222, 3752, 0
};

template < typename Float >
void rgb2hsv ( int R, int G, int B, Float &H, Float &S, Float &V )
  // RGB -> HSV conversion
{
  int m, iv;
  Float r1, g1, b1, vm;

  iv = Max(R,G);
  if ( B > iv ) iv = B;                     // iv = max{R,G,B}
  m = Min(R,G);
  if ( B < m ) m = B;                       // m = min{R,G,B}
  vm = (V = iv) - m;                        // vm = max - min
  S = (iv == 0) ? FLOAT(0) : vm / V;        // Saturation
  if ( S == FLOAT(0) ) H = FLOAT(0);
  else
  {
    vm = FLOAT(1) / vm;
    r1 = (V - R) * vm;
    g1 = (V - G) * vm;
    b1 = (V - B) * vm;
    if ( iv == R )
      if ( m == G ) H = FLOAT(5) + b1;
      else          H = FLOAT(1) - g1;
    else
      if ( iv == G )
        if ( m == B ) H = FLOAT(1) + r1;
        else          H = FLOAT(3) - b1;
      else
        if ( m == R ) H = FLOAT(3) + g1;
        else          H = FLOAT(5) - r1;
    H *= FLOAT(60);                         // Hue is in degrees
  }
}

//--- system time ----------------------------------------

/** System startup time. */
unsigned64 startTime = 0;

double getAppTime ()
  // Application time in seconds
{
  return( 1.e-6 * (int64)(getSystemTime() - startTime) );
}

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#if _MSC_VER < 1300
#  include <largeint.h>
#endif

#pragma warning(disable:4073)
#pragma init_seg(lib)

LARGE_INTEGER hpcFrequency;

bool isHpc = false;

void startSystemTime ()
  // check the high-performance counter
{
  if ( !isHpc )
  {
    LARGE_INTEGER frequency;
    isHpc = (QueryPerformanceFrequency(&frequency) != 0);
    if ( isHpc ) hpcFrequency = frequency;
  }
  startTime = getSystemTime();
}

unsigned getClockFrequency ()
  // clock frequency in Hz
{
  return( isHpc ? (hpcFrequency.HighPart ? MAX_U32 : hpcFrequency.LowPart) : 100 );
}

unsigned64 getSystemTime ()
  // returns actual system time in micro-seconds
{
  if ( isHpc )
  {
    LARGE_INTEGER count;
    if ( QueryPerformanceCounter(&count) )
    {
#if _MSC_VER >= 1300
      LARGE_INTEGER sec;
      LARGE_INTEGER remainder;
      sec.QuadPart = count.QuadPart/hpcFrequency.QuadPart;
      remainder.QuadPart = count.QuadPart%hpcFrequency.QuadPart;
#else
      LARGE_INTEGER remainder;
      LARGE_INTEGER sec = LargeIntegerDivide(count,hpcFrequency,&remainder);
#endif
        // time = sec + (remainder/hpcFrequency)
      return( (unsigned64)sec.QuadPart * 1000000U +
              ((unsigned64)remainder.QuadPart * 1000000U) / (unsigned64)hpcFrequency.QuadPart );
    }
  }
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  return( ((((unsigned64)ft.dwHighDateTime) << 32) + ft.dwLowDateTime + 5) / 10 );
}

#else     // not _WIN32

#include <sys/time.h>

void startSystemTime ()
{
  startTime = getSystemTime();
}

unsigned getClockFrequency ()
  // clock frequency in Hz
{
  return 100;
}

#ifdef HAS_CLOCK_GETTIME

  // "clock_gettime()" is defined:

unsigned64 getSystemTime ()
  // returns actual system time in micro-seconds
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME,&ts);
  return( ts.tv_sec * 1000000ULL + (ts.tv_nsec + 500L) / 1000L );
}

#elif defined(HAS_GETTIMEOFDAY)

  // "gettimeofday()" is defined:

unsigned64 getSystemTime ()
  // returns actual system time in micro-seconds
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return( tv.tv_sec * 1000000ULL + tv.tv_usec );
}

#else     // not HAS_CLOCK_GETTIME

  // neither "clock_gettime()" nor "gettimeofday()" defined:

#  include <sys/timeb.h>

unsigned64 getSystemTime ()
  // returns actual system time in micro-seconds
{
  struct timeb tb;
  ftime(&tb);
  return( 1000 * (tb.millitm + 1000 * (unsigned64)tb.time) );
}

#endif

void sleepMs ( unsigned ms )
{
  struct timeval timeout =
    { ms/1000L, (ms%1000L)*1000L };         // seconds, micro-seconds
  select(0,NULL,NULL,NULL,&timeout);
}

#endif

/** Static system-time initializer class. */
static class Init
{
public:
  Init () { startSystemTime(); }
} sInit INIT_PRIORITY_HIGH;

//--- Math class -----------------------------------------

template < typename Float >
void Math<Float>::matrixMultiply3x3 ( const Float *left, const Float *right, Float *result )
{
    // no asserts
  Float tmp[9];
  if ( left == result )
  {
    memcpy( tmp, left, 9*sizeof(Float) );
    left = tmp;
  }
  Float c0, c1, c2;
  int j;
  for ( j = 0; j < 3; j++ )
  {
    const Float *lptr = left;
    c0 = right[j  ];
    c1 = right[j+3];
    c2 = right[j+6];
    result[j  ] = lptr[0] * c0 + lptr[1] * c1 + lptr[2] * c2;
    lptr += 3;
    result[j+3] = lptr[0] * c0 + lptr[1] * c1 + lptr[2] * c2;
    lptr += 3;
    result[j+6] = lptr[0] * c0 + lptr[1] * c1 + lptr[2] * c2;
  }
}

template < typename Float >
void Math<Float>::vector3TimesMatrix3x3 ( const Float *v, const Float *m, Float *result )
{
    // no asserts
  Float tmp[3];
  if ( v == result )
  {
    tmp[0] = v[0];
    tmp[1] = v[1];
    tmp[2] = v[2];
    v = tmp;
  }
  result[0] = m[0] * v[0] + m[3] * v[1] + m[6] * v[2];
  result[1] = m[1] * v[0] + m[4] * v[1] + m[7] * v[2];
  result[2] = m[2] * v[0] + m[5] * v[1] + m[8] * v[2];
}

template < typename Float >
void Math<Float>::matrix3x3TimesVector3 ( const Float *m, const Float *v, Float *result )
{
    // no asserts
  Float tmp[3];
  if ( v == result )
  {
    tmp[0] = v[0];
    tmp[1] = v[1];
    tmp[2] = v[2];
    v = tmp;
  }
  result[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
  result[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
  result[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

template < typename Float >
void Math<Float>::matrixMultiply4x4 ( const Float *left, const Float *right, Float *result )
{
    // no asserts
  Float tmp[16];
  if ( left == result )
  {
    memcpy( tmp, left, 16*sizeof(Float) );
    left = tmp;
  }
  Float c0, c1, c2, c3;
  int j;
  for ( j = 0; j < 4; j++ )
  {
    const Float *lptr = left;
    c0 = right[j   ];
    c1 = right[j+ 4];
    c2 = right[j+ 8];
    c3 = right[j+12];
    result[j   ] = lptr[0] * c0 + lptr[1] * c1 + lptr[2] * c2 + lptr[3] * c3;
    lptr += 4;
    result[j+ 4] = lptr[0] * c0 + lptr[1] * c1 + lptr[2] * c2 + lptr[3] * c3;
    lptr += 4;
    result[j+ 8] = lptr[0] * c0 + lptr[1] * c1 + lptr[2] * c2 + lptr[3] * c3;
    lptr += 4;
    result[j+12] = lptr[0] * c0 + lptr[1] * c1 + lptr[2] * c2 + lptr[3] * c3;
  }
}

template < typename Float >
void Math<Float>::vector4TimesMatrix4x4 ( const Float *v, const Float *m, Float *result )
{
    // no asserts
  Float tmp[4];
  if ( v == result )
  {
    tmp[0] = v[0];
    tmp[1] = v[1];
    tmp[2] = v[2];
    tmp[3] = v[3];
    v = tmp;
  }
  result[0] = m[ 0] * v[0] + m[ 4] * v[1] + m[ 8] * v[2] + m[12] * v[3];
  result[1] = m[ 1] * v[0] + m[ 5] * v[1] + m[ 9] * v[2] + m[13] * v[3];
  result[2] = m[ 2] * v[0] + m[ 6] * v[1] + m[10] * v[2] + m[14] * v[3];
  result[3] = m[ 3] * v[0] + m[ 7] * v[1] + m[11] * v[2] + m[15] * v[3];
}

template < typename Float >
void Math<Float>::vector3TimesMatrix4x4 ( const Float *v, const Float *m, Float *result )
{
    // no asserts
  Float tmp[3];
  if ( v == result )
  {
    tmp[0] = v[0];
    tmp[1] = v[1];
    tmp[2] = v[2];
    v = tmp;
  }
  result[0] = m[ 0] * v[0] + m[ 4] * v[1] + m[ 8] * v[2] + m[12];
  result[1] = m[ 1] * v[0] + m[ 5] * v[1] + m[ 9] * v[2] + m[13];
  result[2] = m[ 2] * v[0] + m[ 6] * v[1] + m[10] * v[2] + m[14];
}

template < typename Float >
void Math<Float>::matrix4x4TimesVector4 ( const Float *m, const Float *v, Float *result )
{
    // no asserts
  Float tmp[4];
  if ( v == result )
  {
    tmp[0] = v[0];
    tmp[1] = v[1];
    tmp[2] = v[2];
    tmp[3] = v[3];
    v = tmp;
  }
  result[0] = m[ 0] * v[0] + m[ 1] * v[1] + m[ 2] * v[2] + m[ 3] * v[3];
  result[1] = m[ 4] * v[0] + m[ 5] * v[1] + m[ 6] * v[2] + m[ 7] * v[3];
  result[2] = m[ 8] * v[0] + m[ 9] * v[1] + m[10] * v[2] + m[11] * v[3];
  result[3] = m[12] * v[0] + m[13] * v[1] + m[14] * v[2] + m[15] * v[3];
}

template < typename Float >
void Math<Float>::matrix4x4TimesVector3 ( const Float *m, const Float *v, Float *result )
{
    // no asserts
  Float tmp[3];
  if ( v == result )
  {
    tmp[0] = v[0];
    tmp[1] = v[1];
    tmp[2] = v[2];
    v = tmp;
  }
  result[0] = m[ 0] * v[0] + m[ 1] * v[1] + m[ 2] * v[2] + m[ 3];
  result[1] = m[ 4] * v[0] + m[ 5] * v[1] + m[ 6] * v[2] + m[ 7];
  result[2] = m[ 8] * v[0] + m[ 9] * v[1] + m[10] * v[2] + m[11];
}

#define SWAP(a,b)  (tmp=(a),(a)=(b),(b)=tmp)

template < typename Float >
void Math<Float>::matrixTranspose3x3 ( Float *m )
{
    // no asserts
  Float tmp;
  SWAP(m[1],m[3]);
  SWAP(m[2],m[6]);
  SWAP(m[5],m[7]);
}

template < typename Float >
void Math<Float>::matrixTranspose4x4 ( Float *m )
{
    // no asserts
  Float tmp;
  SWAP(m[ 1],m[ 4]);
  SWAP(m[ 2],m[ 8]);
  SWAP(m[ 3],m[12]);
  SWAP(m[ 6],m[ 9]);
  SWAP(m[ 7],m[13]);
  SWAP(m[11],m[14]);
}

#undef SWAP

template < typename Float >
void Math<Float>::quaternionMultiply ( const Float *q, const Float *r, Float *result )
{
    // no asserts
  Float tmp[4];
  if ( q == result )
  {
    tmp[0] = q[0];
    tmp[1] = q[1];
    tmp[2] = q[2];
    tmp[3] = q[3];
    q = tmp;
    if ( r == result )
      r = tmp;
  }
  else
  if ( r == result )
  {
    tmp[0] = r[0];
    tmp[1] = r[1];
    tmp[2] = r[2];
    tmp[3] = r[3];
    r = tmp;
  }
  result[0] = q[1]*r[2] - q[2]*r[1] + r[3]*q[0] + q[3]*r[0];
  result[1] = q[2]*r[0] - q[0]*r[2] + r[3]*q[1] + q[3]*r[1];
  result[2] = q[0]*r[1] - q[1]*r[0] + r[3]*q[2] + q[3]*r[2];
  result[3] = q[3]*r[3] - q[0]*r[0] - q[1]*r[1] - q[2]*r[2];
}

template < typename Float >
void Math<Float>::unitQuaternion ( const Float *axis, Float angle, Float *result )
{
    // no asserts
  double n = DOUBLE(axis[0])*axis[0] + DOUBLE(axis[1])*axis[1] + DOUBLE(axis[2])*axis[2];
  if ( Math<double>::isZero(n) )
  {
    result[0] = result[1] = result[2] = FLOAT(0);
    result[3] = FLOAT(1);
    return;
  }
  n = sin(angle) / sqrt(n);
  result[0] = FLOAT( n * axis[0] );
  result[1] = FLOAT( n * axis[1] );
  result[2] = FLOAT( n * axis[2] );
  result[3] = FLOAT(  cos(angle) );
}

template < typename Float >
void Math<Float>::quaternionRotate3 ( const Float *p, const Float *q, Float *result )
{
    // no asserts
    // result = q * p * ~q
  Float tmp[4];
  tmp[0] = p[0];
  tmp[1] = p[1];
  tmp[2] = p[2];
  tmp[3] = FLOAT(1);    // tmp = p
  Float conj[4];
  conj[0] = -q[0];
  conj[1] = -q[1];
  conj[2] = -q[2];
  conj[3] =  q[3];      // conj = ~q
  quaternionMultiply( tmp, conj, tmp );
  Float res[ 4 ];
  quaternionMultiply( q, tmp, res );
  memcpy( result, res, 3*sizeof(Float) );
}

template < typename Float >
void Math<Float>::quaternionRotate4 ( const Float *p, const Float *q, Float *result )
{
    // no asserts
    // result = q * p * ~q
  Float tmp[4];
  Float conj[4];
  conj[0] = -q[0];
  conj[1] = -q[1];
  conj[2] = -q[2];
  conj[3] =  q[3];      // conj = ~q
  quaternionMultiply(p,conj,tmp);
  quaternionMultiply(q,tmp,result);
}

template < typename Float >
void Math<Float>::quaternionToMatrix3x3 ( const Float *q, Float *result )
{
    // no asserts
  Float xx = q[0] * q[0];
  Float xy = q[0] * q[1];
  Float xz = q[0] * q[2];
  Float xw = q[0] * q[3];
  Float yy = q[1] * q[1];
  Float yz = q[1] * q[2];
  Float yw = q[1] * q[3];
  Float zz = q[2] * q[2];
  Float zw = q[2] * q[3];
  Float tmp;
  tmp = yy + zz;
  *result++ = FLOAT(1) - tmp - tmp;         // m00
  tmp = xy + zw;
  *result++ = tmp + tmp;                    // m01
  tmp = xz - yw;
  *result++ = tmp + tmp;                    // m02
  tmp = xy - zw;
  *result++ = tmp + tmp;                    // m10
  tmp = xx + zz;
  *result++ = FLOAT(1) - tmp - tmp;         // m11
  tmp = yz + xw;
  *result++ = tmp + tmp;                    // m12
  tmp = xz + yw;
  *result++ = tmp + tmp;                    // m20
  tmp = yz - xw;
  *result++ = tmp + tmp;                    // m21
  tmp = xx + yy;
  *result   = FLOAT(1) - tmp - tmp;         // m22
}

template < typename Float >
void Math<Float>::quaternionToMatrix4x4 ( const Float *q, Float *result )
{
    // no asserts
  Float xx = q[0] * q[0];
  Float xy = q[0] * q[1];
  Float xz = q[0] * q[2];
  Float xw = q[0] * q[3];
  Float yy = q[1] * q[1];
  Float yz = q[1] * q[2];
  Float yw = q[1] * q[3];
  Float zz = q[2] * q[2];
  Float zw = q[2] * q[3];
  Float tmp;
  tmp = yy + zz;
  *result++ = FLOAT(1) - tmp - tmp;         // m00
  tmp = xy + zw;
  *result++ = tmp + tmp;                    // m01
  tmp = xz - yw;
  *result++ = tmp + tmp;                    // m02
  *result++ = FLOAT(0);                     // m03
  tmp = xy - zw;
  *result++ = tmp + tmp;                    // m10
  tmp = xx + zz;
  *result++ = FLOAT(1) - tmp - tmp;         // m11
  tmp = yz + xw;
  *result++ = tmp + tmp;                    // m12
  *result++ = FLOAT(0);                     // m13
  tmp = xz + yw;
  *result++ = tmp + tmp;                    // m20
  tmp = yz - xw;
  *result++ = tmp + tmp;                    // m21
  tmp = xx + yy;
  *result++ = FLOAT(1) - tmp - tmp;         // m22
  *result++ = FLOAT(0);                     // m23
  *result++ = FLOAT(0);                     // m30
  *result++ = FLOAT(0);                     // m31
  *result++ = FLOAT(0);                     // m32
  *result   = FLOAT(1);                     // m33
}

template < typename Float >
void Math<Float>::matrix3x3ToQuaternion ( const Float *m, Float *result )
{
    // no asserts
  double x =   m[0] - m[4] - m[8];          // 4x^2 - 1
  double y = - m[0] + m[4] - m[8];          // 4y^2 - 1
  double z = - m[0] - m[4] + m[8];          // 4z^2 - 1
  double w =   m[0] + m[4] + m[8];          // 4w^2 - 1

    // find the gratest value:
  int best =
    (x >= y) ?                              // x >= y
     ( (x >= z) ?
      ( (x >= w) ? 0 : 3 ) :                // z > x >= y
      ( (z >= w) ? 2 : 3 ) ) :              // y > x
     ( (y >= z) ?
      ( (y >= w) ? 1 : 3 ) :                // z > y > x
      ( (z >= w) ? 2 : 3 ) );

    // compute the most stable alternative:
  double k;
  switch ( best )
  {
    case 0:
      x = sqrt(x + 1.0);                    // 2x
      w = (m[5] - m[7]) / x;                // 2w
      k = 1.0 / (w + w);                    // 4w^-1
      result[0] = FLOAT( 0.5 * x );
      result[1] = FLOAT( k * (m[6] - m[2]) );
      result[2] = FLOAT( k * (m[1] - m[3]) );
      break;

    case 1:
      y = sqrt(y + 1.0);                    // 2y
      w = (m[6] - m[2]) / y;                // 2w
      k = 1.0 / (w + w);                    // 4w^-1
      result[0] = FLOAT( k * (m[5] - m[7]) );
      result[1] = FLOAT( 0.5 * y );
      result[2] = FLOAT( k * (m[1] - m[3]) );
      break;

    case 2:
      z = sqrt(z + 1.0);                    // 2z
      w = (m[1] - m[3]) / z;                // 2w
      k = 1.0 / (w + w);                    // 4w^-1
      result[0] = FLOAT( k * (m[5] - m[7]) );
      result[1] = FLOAT( k * (m[6] - m[2]) );
      result[2] = FLOAT( 0.5 * z );
      break;

    default:
      w = sqrt(w + 1.0);                    // 2w
      k = 1.0 / (w + w);                    // 4w^-1
      result[0] = FLOAT( k * (m[5] - m[7]) );
      result[1] = FLOAT( k * (m[6] - m[2]) );
      result[2] = FLOAT( k * (m[1] - m[3]) );
  }

  result[3] = FLOAT( 0.5 * w );
}

template < typename Float >
void Math<Float>::quaternionSlerp ( const Float *q, const Float *r, Float t, Float *result )
{
  Float tmp[4];
    // q . r should be >= 0
  double cosa = q[0]*r[0] + q[1]*r[1] + q[2]*r[2] + q[3]*r[3];
  if ( cosa < 0.0 )
  {
    tmp[0] = -q[0];
    tmp[1] = -q[1];
    tmp[2] = -q[2];
    tmp[3] = -q[3];
    q = tmp;
    cosa = -cosa;
  }
    // are q and r the same?
  if ( cosa > 1.0 - Math<double>::eps() )
  {
    memcpy( result, q, 4*sizeof(Float) );
    return;
  }
    // nontrivial SLERP:
  double k = 1.0 / sqrt( 1.0 - cosa * cosa ); // 1/sin(alpha)
  double alpha = acos( cosa );
  double talpha = t * alpha;
  double kq = k * sin( alpha - talpha );
  double kr = k * sin( talpha );
    // fill the reault array:
  result[0] = FLOAT( kq * q[0] + kr * r[0] );
  result[1] = FLOAT( kq * q[1] + kr * r[1] );
  result[2] = FLOAT( kq * q[2] + kr * r[2] );
  result[3] = FLOAT( kq * q[3] + kr * r[3] );
}

template class Math<float>;
