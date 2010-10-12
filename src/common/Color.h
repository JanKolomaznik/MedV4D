#ifndef _COLOR_H
#define _COLOR_H

#include "common/Vector.h"
#include "common/MathTools.h"

template < typename TElementType >
class RGB: public Vector< TElementType, 3 >
{
public:

	RGB( TElementType r, TElementType g, TElementType b ): Vector< TElementType, 3 >( r, g, b )
	{}
	RGB()
	{}

	TElementType &
	Red()
		{ return this->template StaticGet< 0 >(); }
	TElementType
	Red()const
		{ return this->template StaticGet< 0 >(); }

	TElementType &
	Green()
		{ return this->template StaticGet< 1 >(); }
	TElementType
	Green()const
		{ return this->template StaticGet< 1 >(); }

	TElementType &
	Blue()
		{ return this->template StaticGet< 2 >(); }
	TElementType
	Blue()const
		{ return this->template StaticGet< 2 >(); }

};

template < typename TElementType >
class RGBA: public Vector< TElementType, 4 >
{
public:
	RGBA( TElementType r, TElementType g, TElementType b, TElementType a ): Vector< TElementType, 4 >( r, g, b, a )
	{}
	RGBA()
	{}

	TElementType &
	Red()
		{ return this->template StaticGet< 0 >(); }
	TElementType
	Red()const
		{ return this->template StaticGet< 0 >(); }

	TElementType &
	Green()
		{ return this->template StaticGet< 1 >(); }
	TElementType
	Green()const
		{ return this->template StaticGet< 1 >(); }

	TElementType &
	Blue()
		{ return this->template StaticGet< 2 >(); }
	TElementType
	Blue()const
		{ return this->template StaticGet< 2 >(); }

	TElementType &
	Alpha()
		{ return this->template StaticGet< 3 >(); }
	TElementType
	Alpha()const
		{ return this->template StaticGet< 3 >(); }
};

template < typename TElementType >
class HSV: public Vector< TElementType, 3 >
{
public:
	HSV( TElementType h, TElementType s, TElementType v ): Vector< TElementType, 3 >( h, s, v )
	{}
	HSV()
	{}

	TElementType &
	Hue()
		{ return this->template StaticGet< 0 >(); }
	TElementType
	Hue()const
		{ return this->template StaticGet< 0 >(); }

	TElementType &
	Saturation()
		{ return this->template StaticGet< 1 >(); }
	TElementType
	Saturation()const
		{ return this->template StaticGet< 1 >(); }

	TElementType &
	Value()
		{ return this->template StaticGet< 2 >(); }
	TElementType
	Value()const
		{ return this->template StaticGet< 2 >(); }

};

template < typename TElementType >
class HSVA: public Vector< TElementType, 4 >
{
public:
	HSVA( TElementType h, TElementType s, TElementType v, TElementType a ): Vector< TElementType, 4 >( h, s, v, a )
	{}
	HSVA()
	{}

	TElementType &
	Hue()
		{ return this->template StaticGet< 0 >(); }
	TElementType
	Hue()const
		{ return this->template StaticGet< 0 >(); }

	TElementType &
	Saturation()
		{ return this->template StaticGet< 1 >(); }
	TElementType
	Saturation()const
		{ return this->template StaticGet< 1 >(); }

	TElementType &
	Value()
		{ return this->template StaticGet< 2 >(); }
	TElementType
	Value()const
		{ return this->template StaticGet< 2 >(); }

	TElementType &
	Alpha()
		{ return this->template StaticGet< 3 >(); }
	TElementType
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
		hsv.Hue() = 0.0 + 60.0*(tmpRgb.Green() - tmpRgb.Blue());
		if (hsv.Hue() < 0.0) {
			hsv.Hue() += 360.0;
		}
	} else if ( maxRgb == tmpRgb.Green() ) {
		hsv.Hue() = 120.0 + 60.0*( tmpRgb.Blue() - tmpRgb.Red() );
	} else /* maxRgb == tmpRgb.Blue() */ {
		hsv.Hue() = 240.0 + 60.0*( tmpRgb.Red() - tmpRgb.Green() );
	}
	return hsv;
}





#endif /*_COLOR_H*/
