#ifndef VNL_MATH_H_
#define VNL_MATH_H_

namespace M4D {
namespace Cell {

// abs
inline bool           vnl_math_abs(bool x)          { return x; }
inline unsigned char  vnl_math_abs(unsigned char x) { return x; }
inline unsigned char  vnl_math_abs(signed char x)   { return x < 0 ? -x : x; }
inline unsigned char  vnl_math_abs(char x)          { return (unsigned char)x; }
inline unsigned short vnl_math_abs(short x)         { return x < 0 ? -x : x; }
inline unsigned short vnl_math_abs(unsigned short x){ return x; }
inline unsigned int   vnl_math_abs(int x)           { return x < 0 ? -x : x; }
inline unsigned int   vnl_math_abs(unsigned int x)  { return x; }
inline unsigned long  vnl_math_abs(long x)          { return x < 0L ? -x : x; }
inline unsigned long  vnl_math_abs(unsigned long x) { return x; }
inline float          vnl_math_abs(float x)         { return x < 0.0f ? -x : x; }
inline double         vnl_math_abs(double x)        { return x < 0.0 ? -x : x; }
inline long double    vnl_math_abs(long double x)   { return x < 0.0 ? -x : x; }

// max
inline int           vnl_math_max(int x, int y)                     { return (x > y) ? x : y; }
inline unsigned int  vnl_math_max(unsigned int x, unsigned int y)   { return (x > y) ? x : y; }
inline long          vnl_math_max(long x, long y)                   { return (x > y) ? x : y; }
inline unsigned long vnl_math_max(unsigned long x, unsigned long y) { return (x > y) ? x : y;}
inline float         vnl_math_max(float x, float y)                 { return (x < y) ? y : x; }
inline double        vnl_math_max(double x, double y)               { return (x < y) ? y : x; }

// min
inline int           vnl_math_min(int x, int y)                     { return (x < y) ? x : y; }
inline unsigned int  vnl_math_min(unsigned int x, unsigned int y)   { return (x < y) ? x : y; }
inline long          vnl_math_min(long x, long y)                   { return (x < y) ? x : y; }
inline unsigned long vnl_math_min(unsigned long x, unsigned long y) { return (x < y) ? x : y;}
inline float         vnl_math_min(float x, float y)                 { return (x > y) ? y : x; }
inline double        vnl_math_min(double x, double y)               { return (x > y) ? y : x; }

// sqr (square)
inline bool          vnl_math_sqr(bool x)          { return x; }
inline int           vnl_math_sqr(int x)           { return x*x; }
inline unsigned int  vnl_math_sqr(unsigned int x)  { return x*x; }
inline long          vnl_math_sqr(long x)          { return x*x; }
inline unsigned long vnl_math_sqr(unsigned long x) { return x*x; }
inline float         vnl_math_sqr(float x)         { return x*x; }
inline double        vnl_math_sqr(double x)        { return x*x; }

// cube
inline bool          vnl_math_cube(bool x)          { return x; }
inline int           vnl_math_cube(int x)           { return x*x*x; }
inline unsigned int  vnl_math_cube(unsigned int x)  { return x*x*x; }
inline long          vnl_math_cube(long x)          { return x*x*x; }
inline unsigned long vnl_math_cube(unsigned long x) { return x*x*x; }
inline float         vnl_math_cube(float x)         { return x*x*x; }
inline double        vnl_math_cube(double x)        { return x*x*x; }

// sgn (sign in -1, 0, +1)
inline int vnl_math_sgn(int x)    { return x?((x>0)?1:-1):0; }
inline int vnl_math_sgn(long x)   { return x?((x>0)?1:-1):0; }
inline int vnl_math_sgn(float x)  { return (x != 0)?((x>0)?1:-1):0; }
inline int vnl_math_sgn(double x) { return (x != 0)?((x>0)?1:-1):0; }

// sgn0 (sign in -1, +1 only, useful for reals)
inline int vnl_math_sgn0(int x)    { return (x>=0)?1:-1; }
inline int vnl_math_sgn0(long x)   { return (x>=0)?1:-1; }
inline int vnl_math_sgn0(float x)  { return (x>=0)?1:-1; }
inline int vnl_math_sgn0(double x) { return (x>=0)?1:-1; }

// squared_magnitude
inline unsigned int  vnl_math_squared_magnitude(char          x) { return int(x)*int(x); }
inline unsigned int  vnl_math_squared_magnitude(unsigned char x) { return int(x)*int(x); }
inline unsigned int  vnl_math_squared_magnitude(int           x) { return x*x; }
inline unsigned int  vnl_math_squared_magnitude(unsigned int  x) { return x*x; }
inline unsigned long vnl_math_squared_magnitude(long          x) { return x*x; }
inline unsigned long vnl_math_squared_magnitude(unsigned long x) { return x*x; }
inline float         vnl_math_squared_magnitude(float         x) { return x*x; }
inline double        vnl_math_squared_magnitude(double        x) { return x*x; }
inline long double   vnl_math_squared_magnitude(long double   x) { return x*x; }

// cuberoot
//inline float  vnl_math_cuberoot(float  a) { return float((a<0) ? -vcl_exp(vcl_log(-a)/3) : vcl_exp(vcl_log(a)/3)); }
//inline double vnl_math_cuberoot(double a) { return       (a<0) ? -vcl_exp(vcl_log(-a)/3) : vcl_exp(vcl_log(a)/3); }
//
//// hypotenuse
//inline double      vnl_math_hypot(int         x, int         y) { return vcl_sqrt(double(x*x + y*y)); }
//inline float       vnl_math_hypot(float       x, float       y) { return float( vcl_sqrt(double(x*x + y*y)) ); }
//inline double      vnl_math_hypot(double      x, double      y) { return vcl_sqrt(x*x + y*y); }
//inline long double vnl_math_hypot(long double x, long double y) { return vcl_sqrt(x*x + y*y); }



}  // namespace Cell



}  // namespace M4D

#endif /*VNL_MATH_H_*/
