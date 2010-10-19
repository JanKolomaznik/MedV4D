#ifndef _COLOR_H
#define _COLOR_H

#include "common/Vector.h"
#include "common/MathTools.h"
#undef RGB
#ifdef RGB
	#undef RGB
#endif /*RGB*/

template < typename TChannelType >
class RGB: public Vector< TChannelType, 3 >
{
public:
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

	RGB( TChannelType r, TChannelType g, TChannelType b ): Vector< TChannelType, 3 >( r, g, b )
	{}
	RGB()
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

};

template < typename TChannelType >
class RGBA: public Vector< TChannelType, 4 >
{
public:
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

	HSV( TChannelType h, TChannelType s, TChannelType v ): Vector< TChannelType, 3 >( h, s, v )
	{}
	HSV()
	{}

	TChannelType &
	Hue()
		{ return this->template StaticGet< 0 >(); }
	TChannelType
	Hue()const
		{ return this->template StaticGet< 0 >(); }

	TChannelType &
	Saturation()
		{ return this->template StaticGet< 1 >(); }
	TChannelType
	Saturation()const
		{ return this->template StaticGet< 1 >(); }

	TChannelType &
	Value()
		{ return this->template StaticGet< 2 >(); }
	TChannelType
	Value()const
		{ return this->template StaticGet< 2 >(); }

};

template < typename TChannelType >
class HSVA: public Vector< TChannelType, 4 >
{
public:
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

	HSVA( TChannelType h, TChannelType s, TChannelType v, TChannelType a ): Vector< TChannelType, 4 >( h, s, v, a )
	{}
	HSVA()
	{}

	TChannelType &
	Hue()
		{ return this->template StaticGet< 0 >(); }
	TChannelType
	Hue()const
		{ return this->template StaticGet< 0 >(); }

	TChannelType &
	Saturation()
		{ return this->template StaticGet< 1 >(); }
	TChannelType
	Saturation()const
		{ return this->template StaticGet< 1 >(); }

	TChannelType &
	Value()
		{ return this->template StaticGet< 2 >(); }
	TChannelType
	Value()const
		{ return this->template StaticGet< 2 >(); }

	TChannelType &
	Alpha()
		{ return this->template StaticGet< 3 >(); }
	TChannelType
	Alpha()const
		{ return this->template StaticGet< 3 >(); }
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
	float maxRgb = Max< float, 3>( aRgb );
	float minRgb = Min< float, 3>( aRgb );

	if ( EpsilonTest( maxRgb ) ) {
		hsv.Hue() = hsv.Saturation() = hsv.Value() = 0.0f;
	return hsv;
	} 

	hsv.Value() = maxRgb;

	RGBf tmpRgb = aRgb;
	tmpRgb.Red() /= maxRgb;
	tmpRgb.Green() /= maxRgb;
	tmpRgb.Blue() /= maxRgb;
	maxRgb = Max< float, 3>( tmpRgb );
	minRgb = Min< float, 3>( tmpRgb );
	float diffRgb = maxRgb - minRgb;

	if ( EpsilonTest( diffRgb ) ) {
		hsv.Saturation() = 0.0f;
		hsv.Hue() = 0.0f;
		return hsv;
	} 
	hsv.Saturation() = diffRgb;

	/* Normalize saturation to 1 */
	tmpRgb.Red() = ( tmpRgb.Red() - minRgb ) / diffRgb;
	tmpRgb.Green() = ( tmpRgb.Green() - minRgb ) / diffRgb;
	tmpRgb.Blue() = ( tmpRgb.Blue() - minRgb ) / diffRgb;
	maxRgb = Max< float, 3>( tmpRgb );
	minRgb = Min< float, 3>( tmpRgb );

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





#endif /*_COLOR_H*/
