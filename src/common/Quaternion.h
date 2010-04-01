#ifndef QUATERNION_H
#define QUATERNION_H

#include <boost/math/quaternion.hpp>
#include "common/Vector.h"
#include <cmath>

using   ::boost::math::real;
using   ::boost::math::unreal;
using   ::boost::math::sup;
using   ::boost::math::l1;
using   ::boost::math::abs;
using   ::boost::math::norm;
using   ::boost::math::conj;
using   ::boost::math::exp;
using   ::boost::math::pow;
using   ::boost::math::cos;
using   ::boost::math::sin;
using   ::boost::math::tan;

template< typename T>
class Quaternion: public boost::math::quaternion<T>
{
public:
	typedef T value_type;

	explicit Quaternion(T const & requested_a = TypeTraits<T>::Zero, T const & requested_b = TypeTraits<T>::Zero, T const & requested_c = TypeTraits<T>::Zero, T const & requested_d = TypeTraits<T>::Zero)
		: boost::math::quaternion<T>( requested_a, requested_b, requested_c, requested_d )
	{}

	//explicit Quaternion(::std::complex<T> const & z0, ::std::complex<T> const & z1 = ::std::complex<T>());
	template<typename X> 
	explicit Quaternion(Quaternion<X> const & a_recopier)
		: boost::math::quaternion<T>( a_recopier )
	{}

	Quaternion(boost::math::quaternion<T> const & a_recopier)
		: boost::math::quaternion<T>( a_recopier )
	{}

	explicit Quaternion( const Vector<T,3>  & a_vector)
		: boost::math::quaternion<T>( TypeTraits<T>::Zero, a_vector[0], a_vector[1], a_vector[2] )
	{}

	explicit Quaternion( const T &requested_a, const Vector<T,3>  & a_vector)
		: boost::math::quaternion<T>( requested_a, a_vector[0], a_vector[1], a_vector[2] )
	{}

	template<typename X>	
   	Quaternion<T>& operator = (boost::math::quaternion<X> const  & a_affecter)
		{
			boost::math::quaternion<T>::operator=( a_affecter );
			return *this; 
		}

	Vector<T,3>  unreal_vector()
		{
			return Vector<T,3>( this->R_component_2(), this->R_component_3(), this->R_component_4() );
		}

protected:

};

template< typename T >
Quaternion< T >
CreateRotationQuaternion( T angle, const Vector< T, 3 > &axis )
{
	T cangle = cos( 0.5 * angle );
	T sangle = sin( 0.5 * angle );
	return Quaternion< T >( cangle, sangle * axis );
}

template< typename T >
Vector< T, 3 >
RotatePoint( const Vector< T, 3 > &point, const Quaternion< T > &q, const Quaternion< T > &qInv )
{
	Quaternion< T > tmp = q * Quaternion< T >(point) * qInv;

	return tmp.unreal_vector();
}

template< typename T >
Vector< T, 3 >
RotatePoint( const Vector< T, 3 > &point, const Quaternion< T > &q, const boost::math::quaternion<T> &qInv )
{
	Quaternion< T > tmp = q * Quaternion< T >(point) * qInv;

	return tmp.unreal_vector();
}


template< typename T >
Vector< T, 3 >
RotatePoint( const Vector< T, 3 > &point, const Quaternion< T > &q )
{
	

	return RotatePoint( point, q, conj( q ) );
}


/**
 * \param axis Rotation axis direction - must be unit vector
 **/
template< typename T >
Vector< T, 3 >
RotatePoint( const Vector< T, 3 > &point, T angle, const Vector< T, 3 > &axis )
{
	Quaternion< T > tmp( CreateRotationQuaternion( angle, axis ) );
	return RotatePoint( point, tmp );
	
}


#endif /*QUATERNION_H*/
