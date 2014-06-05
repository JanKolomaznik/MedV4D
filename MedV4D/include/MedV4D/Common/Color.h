#ifndef _COLOR_H
#define _COLOR_H

#include "MedV4D/Common/Vector.h"
#include "MedV4D/Common/MathTools.h"
#undef RGB
#ifdef RGB
	#undef RGB
#endif /*RGB*/

namespace M4D
{

template < typename TChannelType >
class RGB: public Vector< TChannelType, 3 >
{
public:
	typedef Vector< TChannelType, 3 > Predecessor;
	template< unsigned tCoord >
	struct ValueAccessor
	{
		TChannelType &
		operator()( RGB< TChannelType > & aData ) const
		{
			return aData.template StaticGet< tCoord >();
		}
		TChannelType
		operator()( const RGB< TChannelType > & aData ) const
		{
			return aData.template StaticGet< tCoord >();
		}
	};

	RGB( const Predecessor &p ): Predecessor ( p )
	{}

	RGB( TChannelType r, TChannelType g, TChannelType b ): Vector< TChannelType, 3 >( r, g, b )
	{}
	RGB()
	{}

	TChannelType &
	Red()
		{ return this->template staticGet< 0 >(); }
	TChannelType
	Red()const
		{ return this->template staticGet< 0 >(); }

	TChannelType &
	Green()
		{ return this->template staticGet< 1 >(); }
	TChannelType
	Green()const
		{ return this->template staticGet< 1 >(); }

	TChannelType &
	Blue()
		{ return this->template staticGet< 2 >(); }
	TChannelType
	Blue()const
		{ return this->template staticGet< 2 >(); }

};

template < typename TChannelType >
class RGBA: public Vector< TChannelType, 4 >
{
public:
	typedef Vector< TChannelType, 4 > Predecessor;
	template< unsigned tCoord >
	struct ValueAccessor
	{
		TChannelType &
		operator()( RGBA< TChannelType > & aData ) const
		{
			return aData.template StaticGet< tCoord >();
		}
		TChannelType
		operator()( const RGBA< TChannelType > & aData ) const
		{
			return aData.template StaticGet< tCoord >();
		}
	};

	RGBA( const Predecessor &p ): Predecessor ( p )
	{}

	RGBA( TChannelType r, TChannelType g, TChannelType b, TChannelType a ): Vector< TChannelType, 4 >( r, g, b, a )
	{}
	RGBA()
	{}

	TChannelType &
	Red()
		{ return this->template StaticGet< 0 >(); }
	TChannelType
	Red()const
		{ return this->template StaticGet< 0 >(); }

	TChannelType &
	Green()
		{ return this->template StaticGet< 1 >(); }
	TChannelType
	Green()const
		{ return this->template StaticGet< 1 >(); }

	TChannelType &
	Blue()
		{ return this->template StaticGet< 2 >(); }
	TChannelType
	Blue()const
		{ return this->template StaticGet< 2 >(); }

	TChannelType &
	Alpha()
		{ return this->template StaticGet< 3 >(); }
	TChannelType
	Alpha()const
		{ return this->template StaticGet< 3 >(); }
};

template < typename TChannelType >
class HSV: public Vector< TChannelType, 3 >
{
public:
	typedef Vector< TChannelType, 3 > Predecessor;
	template< unsigned tCoord >
	struct ValueAccessor
	{
		TChannelType &
		operator()( HSV< TChannelType > & aData ) const
		{
			return aData.template StaticGet< tCoord >();
		}
		TChannelType
		operator()( const HSV< TChannelType > & aData ) const
		{
			return aData.template StaticGet< tCoord >();
		}
	};

	HSV( const Predecessor &p ): Predecessor ( p )
	{}

	HSV( TChannelType h, TChannelType s, TChannelType v ): Vector< TChannelType, 3 >( h, s, v )
	{}
	HSV()
	{}

	TChannelType &
	Hue()
		{ return this->template staticGet< 0 >(); }
	TChannelType
	Hue()const
		{ return this->template staticGet< 0 >(); }

	TChannelType &
	Saturation()
		{ return this->template staticGet< 1 >(); }
	TChannelType
	Saturation()const
		{ return this->template staticGet< 1 >(); }

	TChannelType &
	Value()
		{ return this->template staticGet< 2 >(); }
	TChannelType
	Value()const
		{ return this->template staticGet< 2 >(); }

};

template < typename TChannelType >
class HSVA: public Vector< TChannelType, 4 >
{
public:
	typedef Vector< TChannelType, 4 > Predecessor;
	template< unsigned tCoord >
	struct ValueAccessor
	{
		TChannelType &
		operator()( HSVA< TChannelType > & aData ) const
		{
			return aData.template StaticGet< tCoord >();
		}
		TChannelType
		operator()( const HSVA< TChannelType > & aData ) const
		{
			return aData.template StaticGet< tCoord >();
		}
	};

	HSVA( const Predecessor &p ): Predecessor ( p )
	{}

	HSVA( TChannelType h, TChannelType s, TChannelType v, TChannelType a ): Vector< TChannelType, 4 >( h, s, v, a )
	{}
	HSVA()
	{}

	TChannelType &
	Hue()
		{ return this->template staticGet< 0 >(); }
	TChannelType
	Hue()const
		{ return this->template staticGet< 0 >(); }

	TChannelType &
	Saturation()
		{ return this->template staticGet< 1 >(); }
	TChannelType
	Saturation()const
		{ return this->template staticGet< 1 >(); }

	TChannelType &
	Value()
		{ return this->template staticGet< 2 >(); }
	TChannelType
	Value()const
		{ return this->template staticGet< 2 >(); }

	TChannelType &
	Alpha()
		{ return this->template staticGet< 3 >(); }
	TChannelType
	Alpha()const
		{ return this->template staticGet< 3 >(); }
};

typedef RGB< uint8 > RGB_byte;
typedef RGB< float32 > RGB_float;

typedef RGBA< uint8 > RGBA_byte;
typedef RGBA< float32 > RGBA_float;

typedef RGB< float > RGBf;
typedef RGB< double > RGBd;

typedef RGBA< float > RGBAf;
typedef RGBA< double > RGBAd;

typedef HSV< float > HSVf;
typedef HSV< double > HSVd;

typedef HSVA< float > HSVAf;
typedef HSVA< double > HSVAd;


template< typename TColor >
inline bool
ValidateColor( const TColor &aColor )
{
	//TODO test bands
	return true;
}

inline HSVf
RgbToHsv( const RGBf &aRgb )
{
	ASSERT( ValidateColor( aRgb ) );

	HSVf hsv;
	float maxRgb = max< float, 3>( aRgb );
	float minRgb = min< float, 3>( aRgb );

	if ( epsilonTest( maxRgb ) ) {
		hsv.Hue() = hsv.Saturation() = hsv.Value() = 0.0f;
	return hsv;
	}

	hsv.Value() = maxRgb;

	RGBf tmpRgb = aRgb;
	tmpRgb.Red() /= maxRgb;
	tmpRgb.Green() /= maxRgb;
	tmpRgb.Blue() /= maxRgb;
	maxRgb = max< float, 3>( tmpRgb );
	minRgb = min< float, 3>( tmpRgb );
	float diffRgb = maxRgb - minRgb;

	if ( epsilonTest( diffRgb ) ) {
		hsv.Saturation() = 0.0f;
		hsv.Hue() = 0.0f;
		return hsv;
	}
	hsv.Saturation() = diffRgb;

	/* Normalize saturation to 1 */
	tmpRgb.Red() = ( tmpRgb.Red() - minRgb ) / diffRgb;
	tmpRgb.Green() = ( tmpRgb.Green() - minRgb ) / diffRgb;
	tmpRgb.Blue() = ( tmpRgb.Blue() - minRgb ) / diffRgb;
	maxRgb = max< float, 3>( tmpRgb );
	minRgb = min< float, 3>( tmpRgb );

	/* Compute hue */
	if (maxRgb == tmpRgb.Red() ) {
		hsv.Hue() = 0.0f + 60.0f*(tmpRgb.Green() - tmpRgb.Blue());
		if (hsv.Hue() < 0.0f) {
			hsv.Hue() += 360.0f;
		}
	} else if ( maxRgb == tmpRgb.Green() ) {
		hsv.Hue() = 120.0f + 60.0f*( tmpRgb.Blue() - tmpRgb.Red() );
	} else /* maxRgb == tmpRgb.Blue() */ {
		hsv.Hue() = 240.0f + 60.0f*( tmpRgb.Red() - tmpRgb.Green() );
	}
	return hsv;
}


template < typename TChannelType >
RGBA< TChannelType >
operator+( const RGBA< TChannelType > &c1, const RGBA< TChannelType > &c2 )
{
	return RGBA< TChannelType >( c1.Red() + c2.Red(), c1.Green() + c2.Green(), c1.Blue() + c2.Blue(),  c1.Alpha() + c2.Alpha() );
}

template < typename TChannelType >
RGBA< TChannelType >
operator-( const RGBA< TChannelType > &c1, const RGBA< TChannelType > &c2 )
{
	return RGBA< TChannelType >( static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1)
		- static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1) );
}

template < typename TChannelType >
RGBA< TChannelType >
operator+=( const RGBA< TChannelType > &c1, const RGBA< TChannelType > &c2 )
{
	return RGBA< TChannelType >( static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1)
		+= static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1) );
}

template < typename TChannelType >
RGBA< TChannelType >
operator-=( const RGBA< TChannelType > &c1, const RGBA< TChannelType > &c2 )
{
	return RGBA< TChannelType >( static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1)
		-= static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1) );
}

/*template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
operator*( CoordType k, const Vector< CoordType, Dim > &v )
{
	return static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1)
		+ static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1);
}

template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
operator*( const Vector< CoordType, Dim > &v, CoordType k )
{
	return static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1)
		+ static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1);
}

template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
operator*=( Vector< CoordType, Dim > &v, CoordType k )
{
	return static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1)
		+ static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1);
}

template< typename CoordType, size_t Dim >
CoordType
operator*( const Vector< CoordType, Dim > &a, const Vector< CoordType, Dim > &b )
{
	return static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1)
		+ static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1);
}

template< typename CoordType, size_t Dim >
Vector< CoordType, Dim >
operator-( const Vector< CoordType, Dim > &v )
{
	return static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1)
		+ static_cast< const typename RGBA< TChannelType >::Predecessor & > (c1);
}*/



}//namespace M4D


#endif /*_COLOR_H*/
