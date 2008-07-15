#ifdef _MSC_VER
#  pragma once
#endif

/**
 *  vector.h - 3D, 4D vectors.
 *  License: General Public License GPL (http://www.gnu.org/copyleft/gpl.html)
 *
 *  @author  Josef Pelikan $Author: pepca $
 *  @version $Rev: 82 $
 *  @date    $Date: 2008-05-03 13:46:10 +0200 (Sat, 03 May 2008) $
 */
#ifndef _VECTOR_H
#define _VECTOR_H

#include "GUI/ogl/base.h"

template < typename Float >
class Vector4;

template < typename Float >
class Matrix3x3;

template < typename Float >
class Matrix4x4;

template < typename Float >
class Quaternion;

/**
 *  3-dimensional vector.
 */
template < typename Float >
class Vector3
{
  friend class Vector4<Float>;
  friend class Matrix3x3<Float>;
  friend class Matrix4x4<Float>;
  friend class Quaternion<Float>;

protected:

  typedef Math<Float>   MyMath;

  Float v[3];

public:

  //--- constructions --------------------------------------

  inline Vector3 ()
  {}

  inline Vector3 ( Float x,
                   Float y = FLOAT(0),
                   Float z = FLOAT(0) )
  {
    v[0] = x; v[1] = y; v[2] = z;
  }

  inline Vector3 ( const Float *arr )
  {
    Assert( arr );
    memcpy( v, arr, 3*sizeof(Float) );
  }

  inline Vector3 ( const Vector3 &from )
  {
    memcpy( v, from.v, 3*sizeof(Float) );
  }

  inline Vector3 ( const Vector3 &start, const Vector3 &end )
  {
    v[0] = end.v[0] - start.v[0];
    v[1] = end.v[1] - start.v[1];
    v[2] = end.v[2] - start.v[2];
  }

  Vector3 ( const Vector4<Float> &from )
  {
    if ( from.isFinite() )
    {
      double k = 1.0 / from[3];
      v[0] = FLOAT( k * from[0] );
      v[1] = FLOAT( k * from[1] );
      v[2] = FLOAT( k * from[2] );
    }
    else
    {
      v[0] = from[0];
      v[1] = from[1];
      v[2] = from[2];
    }
  }

  Vector3 ( const Quaternion<Float> &q )
  {
    if ( MyMath::isZero(q[3]) )
    {
      v[0] = q[0]; v[1] = q[1]; v[2] = q[2];
    }
    else
    {
      double k = 1.0 / q[3];
      v[0] = FLOAT( k * q[0] );
      v[1] = FLOAT( k * q[1] );
      v[2] = FLOAT( k * q[2] );
    }
  }

  //--- operations -----------------------------------------

  inline Vector3& operator = ( const Vector3 &from )
  {
    memcpy( v, from.v, 3*sizeof(Float) );
    return *this;
  }

  inline Float operator [] ( int i ) const
  {
    Assert( i >= 0 && i < 3 );
    return v[i];
  }

  inline Float& operator [] ( int i )
  {
    Assert( i >= 0 && i < 3 );
    return v[i];
  }

  inline Float* data ()
  {
    return v;
  }

  inline void get ( Float *arr ) const
  {
    Assert( arr );
    memcpy( arr, v, 3*sizeof(Float) );
  }

  inline void set ( Float x, Float y, Float z )
  {
    v[0] = x; v[1] = y; v[2] = z;
  }

  inline void set ( const Float *arr )
  {
    Assert( arr );
    memcpy( v, arr, 3*sizeof(Float) );
  }

  inline Vector3 operator + ( const Vector3 &add ) const
  {
    return Vector3( v[0] + add.v[0],
                    v[1] + add.v[1],
                    v[2] + add.v[2] );
  }

  inline Vector3& operator += ( const Vector3 &add )
  {
    v[0] += add.v[0];
    v[1] += add.v[1];
    v[2] += add.v[2];
    return *this;
  }

  inline Vector3 operator - ( const Vector3 &sub ) const
  {
    return Vector3(sub,*this);
  }

  inline Vector3& operator -= ( const Vector3 &sub )
  {
    v[0] -= sub.v[0];
    v[1] -= sub.v[1];
    v[2] -= sub.v[2];
    return *this;
  }

  inline Vector3 operator -() const
  {
    return Vector3( -v[0], -v[1], -v[2] );
  }

  inline void reverse ()
  {
    v[0] = -v[0];
    v[1] = -v[1];
    v[2] = -v[2];
  }

  Vector3 cross ( const Vector3 &p ) const
  {
    return Vector3( v[1]*p.v[2] - v[2]*p.v[1],
                    v[2]*p.v[0] - v[0]*p.v[2],
                    v[0]*p.v[1] - v[1]*p.v[0] );
  }

  inline Float operator * ( const Vector3 &d ) const
  {
    return( v[0]*d.v[0] + v[1]*d.v[1] + v[2]*d.v[2] );
  }

  inline Vector3 operator * ( Float k ) const
  {
    return Vector3( k * v[0],
                    k * v[1],
                    k * v[2] );
  }

  inline Vector3& operator *= ( Float k )
  {
    v[0] *= k;
    v[1] *= k;
    v[2] *= k;
    return *this;
  }

  inline Vector3 operator * ( const Matrix3x3<Float> &m ) const
  {
    Float tmp[3];
    MyMath::vector3TimesMatrix3x3(v,m.m,tmp);
    return Vector3(tmp);
  }

  inline Vector3& operator *= ( const Matrix3x3<Float> &m )
  {
    MyMath::vector3TimesMatrix3x3(v,m.m,v);
    return *this;
  }

  inline Vector3 operator * ( const Matrix4x4<Float> &m ) const
  {
    Float tmp[3];
    MyMath::vector3TimesMatrix4x4(v,m.m,tmp);
    return Vector3(tmp);
  }

  inline Vector3& operator *= ( const Matrix4x4<Float> &m )
  {
    MyMath::vector3TimesMatrix4x4(v,m.m,v);
    return *this;
  }

  inline Vector3 operator * ( const Quaternion<Float> &r ) const
  {
    Float tmp[3];
    MyMath::quaternionRotate3(v,r.u,tmp);
    return Vector3(tmp);
  }

  inline Vector3& operator *= ( const Quaternion<Float> &r )
  {
    MyMath::quaternionRotate3(v,r.u,v);
    return *this;
  }

  inline Float size () const
  {
    return FLOAT(MyMath::distance3(v));
  }

  inline double sizeD () const
  {
    return MyMath::distance3(v);
  }

  inline Float normalize ()
  {
    return MyMath::normalize3(v);
  }

};

/**
 *  4-dimensional (homogenous) vector.
 */
template < typename Float >
class Vector4
{
  friend class Vector3<Float>;
  friend class Matrix3x3<Float>;
  friend class Matrix4x4<Float>;
  friend class Quaternion<Float>;

protected:

  typedef Math<Float>   MyMath;

  Float v[4];

public:

  //--- constructions --------------------------------------

  inline Vector4 ()
  {}

  inline Vector4 ( Float x,
                   Float y = FLOAT(0),
                   Float z = FLOAT(0),
                   Float w = FLOAT(1) )
  {
    v[0] = x; v[1] = y; v[2] = z; v[3] = w;
  }

  inline Vector4 ( const Float *arr )
  {
    Assert( arr );
    memcpy( v, arr, 4*sizeof(Float) );
  }

  inline Vector4 ( const Vector4 &from )
  {
    memcpy( v, from.v, 4*sizeof(Float) );
  }

  inline Vector4 ( const Vector3<float> &from )
  {
    memcpy( v, from.v, 3*sizeof(Float) );
    v[3] = FLOAT(1);
  }

  inline Vector4 ( const Vector3<float> &start, const Vector3<float> &end )
  {
    v[0] = end.v[0] - start.v[0];
    v[1] = end.v[1] - start.v[1];
    v[2] = end.v[2] - start.v[2];
    v[3] = FLOAT(0);
  }

  inline Vector4 ( const Quaternion<Float> &q )
  {
    memcpy( v, q.u, 4*sizeof(Float) );
  }

  //--- operations -----------------------------------------

  inline Vector4& operator = ( const Vector4 &from )
  {
    memcpy( v, from.v, 4*sizeof(Float) );
    return *this;
  }

  inline Float operator [] ( int i ) const
  {
    Assert( i >= 0 && i < 4 );
    return v[i];
  }

  inline Float& operator [] ( int i )
  {
    Assert( i >= 0 && i < 4 );
    return v[i];
  }

  inline void get ( Float *arr ) const
  {
    Assert( arr );
    memcpy( arr, v, 4*sizeof(Float) );
  }

  inline void set ( const Float *arr )
  {
    Assert( arr );
    memcpy( v, arr, 4*sizeof(Float) );
  }

  inline bool isFinite () const
  {
    return( !MyMath::isZero(v[3]) );
  }

  inline Vector4 operator + ( const Vector4 &add ) const
  {
    return Vector4( v[0] + add.v[0],
                    v[1] + add.v[1],
                    v[2] + add.v[2],
                    v[3] + add.v[3] );
  }

  inline Vector4& operator += ( const Vector4 &add )
  {
    v[0] += add.v[0];
    v[1] += add.v[1];
    v[2] += add.v[2];
    v[3] += add.v[3];
    return *this;
  }

  inline Vector4 operator - ( const Vector4 &sub ) const
  {
    return Vector4( v[0] - sub.v[0],
                    v[1] - sub.v[1],
                    v[2] - sub.v[2],
                    v[3] - sub.v[3] );
  }

  inline Vector4& operator -= ( const Vector4 &sub )
  {
    v[0] -= sub.v[0];
    v[1] -= sub.v[1];
    v[2] -= sub.v[2];
    v[3] -= sub.v[3];
    return *this;
  }

  inline Vector4 operator -() const
  {
    return Vector4( -v[0], -v[1], -v[2], -v[3] );
  }

  inline void reverse ()
  {
    v[0] = -v[0];
    v[1] = -v[1];
    v[2] = -v[2];
  }

  Vector4 cross ( const Vector4 &p ) const
  {
    return Vector4( v[1]*p.v[2] - v[2]*p.v[1],
                    v[2]*p.v[0] - v[0]*p.v[2],
                    v[0]*p.v[1] - v[1]*p.v[0],
                    v[3]*p.v[3] );
  }

  inline Float operator * ( const Vector4 &d ) const
  {
    return( v[0]*d.v[0] + v[1]*d.v[1] + v[2]*d.v[2] + v[3]*d.v[3] );
  }

  inline Vector4 operator * ( Float k ) const
  {
    return Vector4( k * v[0],
                    k * v[1],
                    k * v[2],
                        v[3] );     // !!! not sure !!!
  }

  inline Vector4& operator *= ( Float k )
  {
    v[0] *= k;
    v[1] *= k;
    v[2] *= k;
    return *this;
  }

  inline Vector4 operator * ( const Matrix4x4<Float> &m ) const
  {
    Float tmp[4];
    MyMath::vector4TimesMatrix4x4(v,m.m,tmp);
    return Vector4(tmp);
  }

  inline Vector4& operator *= ( const Matrix4x4<Float> &m )
  {
    MyMath::vector4TimesMatrix4x4(v,m.m,v);
    return *this;
  }

  inline Vector4 operator * ( const Quaternion<Float> &r ) const
  {
    Float tmp[4];
    MyMath::quaternionRotate4(v,r.u,tmp);
    return Vector4(tmp);
  }

  inline Vector4& operator *= ( const Quaternion<Float> &r )
  {
    MyMath::quaternionRotate4(v,r.u,v);
    return *this;
  }

  inline Float size () const
  {
    return FLOAT(MyMath::distance4(v));
  }

  inline double sizeD () const
  {
    return MyMath::distance4(v);
  }

  inline Float normalize ()
  {
    return MyMath::normalize4(v);
  }

};

#endif
