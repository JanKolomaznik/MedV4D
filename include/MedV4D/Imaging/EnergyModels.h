#ifndef ENERGY_MODELS_H
#define ENERGY_MODELS_H

#include "Imaging/PointSet.h"
#include "Imaging/BSpline.h"
#include <vector>

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file EnergyModels.h 
 * @{ 
 **/

namespace Imaging
{
namespace Algorithms
{

template< typename BufferType, typename SamplesSet, typename RegionType >
void
FillSampleValuesBufferFromRegion( BufferType &buffer, const SamplesSet & samples, RegionType &region )
{
	int32 sampleCount = samples.Size();
	buffer.resize( sampleCount );
	for( int32 i = 0; i < sampleCount; ++i ) {
		buffer[ i ] = region.GetElementWorldCoords( samples[i] );
	}
}

template< typename ContourType, typename ImageEnergy, typename InternalEnergy, typename ConstrainEnergy >
class SegmentationEnergy: public ImageEnergy, public InternalEnergy, public ConstrainEnergy
{
public:
	typedef M4D::Imaging::Geometry::PointSet< typename ContourType::PointType > 	GradientType;
	typedef typename ContourType::PointType						PointCoordinate;

	SegmentationEnergy(): 
			_imageEnergyBalance( 1.0f ),
			_internalEnergyBalance( 1.0f ),
			_constrainEnergyBalance( 1.0f )
		{}
	void
	ResetEnergy() {}

	float32
	GetParametersGradient( ContourType &curve, GradientType &gradient )
	{
		static const float32 Epsilon = 0.0001f;

		if( curve.Size() != gradient.Size() ) {
			//TODO - solve problem
		}

		GradientType * gradients[3];
		unsigned gradientCount = 0;
		float32 gradientNorms[3];

		bool doImageGradient = abs(_imageEnergyBalance) > Epsilon;
		bool doInternalGradient = abs(_internalEnergyBalance) > Epsilon;
		bool doConstrainGradient = abs(_constrainEnergyBalance) > Epsilon;
		//******** ImageEnergy ***********************************
		GradientType imageEnergyGradient;
		float32 imageEnergyGradientNorm = 0.0f;
		if( doImageGradient ) {
			imageEnergyGradient.Resize( gradient.Size() );
			imageEnergyGradientNorm = ImageEnergy::GetParametersGradient( curve, imageEnergyGradient );

			//check if gradient is considerable and set normalization factor multiplied by balance
			if( (doImageGradient = (abs(imageEnergyGradientNorm) > Epsilon)) ) { 
				imageEnergyGradientNorm = _imageEnergyBalance / imageEnergyGradientNorm;

				gradientNorms[ gradientCount ] = imageEnergyGradientNorm;
				gradients[ gradientCount++ ] = &imageEnergyGradient;
			}
		}
		
		//******** InternalEnergy *********************************
		GradientType internalEnergyGradient;
		float32 internalEnergyGradientNorm = 0.0f;
		if( doInternalGradient ) {
			internalEnergyGradient.Resize( gradient.Size() );
			internalEnergyGradientNorm = InternalEnergy::GetParametersGradient( curve, internalEnergyGradient );
		
			//check if gradient is considerable and set normalization factor multiplied by balance
			if( (doInternalGradient = (abs(internalEnergyGradientNorm) > Epsilon)) ) { 
				internalEnergyGradientNorm = _internalEnergyBalance / internalEnergyGradientNorm;

				gradientNorms[ gradientCount ] = internalEnergyGradientNorm;
				gradients[ gradientCount++ ] = &internalEnergyGradient;
			}
		}

		//******** ConstrainEnergy ********************************
		GradientType constrainEnergyGradient;
		float32 constrainEnergyGradientNorm = 0.0f;
		if( doConstrainGradient ) {
			constrainEnergyGradient.Resize( gradient.Size() );
			constrainEnergyGradientNorm = InternalEnergy::GetParametersGradient( curve, constrainEnergyGradient );

			//check if gradient is considerable and set normalization factor multiplied by balance
			if( (doConstrainGradient = (abs(constrainEnergyGradientNorm) > Epsilon)) ) { 
				constrainEnergyGradientNorm = _constrainEnergyBalance / constrainEnergyGradientNorm;

				gradientNorms[ gradientCount ] = constrainEnergyGradientNorm;
				gradients[ gradientCount++ ] = &constrainEnergyGradient;
			}
		}

		//Consolidate all gradients into one
		float32 resultSize = 0.0f;

		for( unsigned i = 0; i < gradient.Size(); ++i ) {
			PointCoordinate point;
			for( unsigned j = 0; j < gradientCount; ++j ) {
				point += gradientNorms[j] * (*gradients[j])[i];
			}
			
			gradient[i] = point;
			resultSize += point * point;
		}
		return sqrt(resultSize);
	}

	float32
	GetImageEnergyBalance()const
		{ return _imageEnergyBalance; }

	void
	SetImageEnergyBalance( float32 value )
		{ _imageEnergyBalance = value; }

	float32
	GetInternalEnergyBalance()const
		{ return _internalEnergyBalance; }

	void
	SetInternalEnergyBalance( float32 value )
		{ _internalEnergyBalance = value; }

	float32
	GetConstrainEnergyBalance()const
		{ return _constrainEnergyBalance; }

	void
	SetConstrainEnergyBalance( float32 value )
		{ _constrainEnergyBalance = value; }
private:
	float32	_imageEnergyBalance;
	float32	_internalEnergyBalance;
	float32	_constrainEnergyBalance;

};

template< typename ContourType, typename RegionType >
class GradientMagnitudeEnergy
{
public:
	typedef M4D::Imaging::Geometry::PointSet< typename ContourType::PointType > 	GradientType;
	typedef typename ContourType::PointType						PointCoordinate;
	typedef typename RegionType::ElementType					ElementType;
	typedef typename ContourType::SamplePointSet					SamplePointSet;
	typedef std::vector< ElementType >						ValuesAtSamplesBuffer;

	void
	ResetEnergy() {}

	float32
	GetParametersGradient( ContourType &curve, GradientType &gradient )
	{
		if( curve.Size() != gradient.Size() ) {
			//TODO - solve problem
		}
		
		float32 gradSize = 0.0f;
		const SamplePointSet &samples = curve.GetSamplePoints();
		int32 sampleFrequency = curve.GetLastSampleFrequency();
		int32 sampleCount = samples.Size();
		static const int32 degree = ContourType::CurveBasis::Degree;
		//int32 segmentCount = curve.GetSegmentCount();
		
		//fill buffer with gradients on curve
		FillSampleValuesBufferFromRegion( _valBuffer, samples, _region );

		for( int32 i = 0; i < (int32)gradient.Size(); ++i ) {
			gradient[i] = PointCoordinate();
			
			for( int32 j = -degree; j < 1; ++j ) {
				for( int32 idx = 0; idx < sampleFrequency; ++idx ) {
					int pIdx = (i + j)*sampleFrequency + idx;
					if( pIdx < 0 ) pIdx += sampleCount;
					if( pIdx >= sampleCount ) pIdx -= sampleCount;

					//Get values for right t and point
					float coef = curve.GetLastBasisFunctionValues()[ idx ][ -1 * j ];
					gradient[i][0] += coef * _valBuffer[ pIdx ].data[0];
					gradient[i][1] += coef * _valBuffer[ pIdx ].data[1];
				}
			}
			gradSize += gradient[i] * gradient[i];
		}
		return sqrt( gradSize );
	}
	
	void
	SetRegionEdge( const RegionType &region )
		{ _region = region; }
private:
	RegionType		_region;
	ValuesAtSamplesBuffer	_valBuffer;
};

template< typename ContourType, typename RegionType1, typename RegionType2, typename Distribution >
class UnifiedImageEnergy : public Distribution
{
public:
	typedef  M4D::Imaging::Geometry::PointSet< typename ContourType::PointType > 	GradientType;
	typedef typename ContourType::PointType	PointCoordinate;
	typedef typename ContourType::BFValVector	BFValVector;
	//typedef typename RegionType1::ElementType	ElementType1;
	//typedef typename RegionType2::ElementType	ElementType2;
	typedef std::vector< float32 >			ValuesAtSamplesBuffer;
	typedef typename ContourType::SamplePointSet	SampleSet;
	static const unsigned Degree = ContourType::Degree;

	UnifiedImageEnergy() : _alpha(0.5f), _sampleFrequency( -1 )
		{}

	void
	ResetEnergy() {}

	
	float32
	GetParametersGradient( ContourType &curve, GradientType &gradient )
	{
		if( curve.Size() != gradient.Size() ) {
			//TODO - solve problem
		}

		if( _sampleFrequency != (int32)curve.GetLastSampleFrequency() ) {
			RecalculateQki( curve );
		}

		FillSampleValuesBuffers( curve.GetSamplePoints() );		

		GradientType regionEnergyGradient;
		regionEnergyGradient.Resize( gradient.Size() );
		float32 regionEnergyGradientNorm = 0.0f;

		GradientType edgeEnergyGradient;
		edgeEnergyGradient.Resize( gradient.Size() );
		float32 edgeEnergyGradientNorm = 0.0f;

		for( unsigned i = 0; i < gradient.Size(); ++i ) {
			regionEnergyGradient[i] = ComputePointGradient( i, curve, _regionValBuffer );
			regionEnergyGradientNorm += regionEnergyGradient[i] * regionEnergyGradient[i];

			edgeEnergyGradient[i] = ComputePointGradient( i, curve, _edgeValBuffer );
			edgeEnergyGradientNorm += edgeEnergyGradient[i] * edgeEnergyGradient[i];
		}
		regionEnergyGradientNorm = _alpha * 1.0f/regionEnergyGradientNorm;
		edgeEnergyGradientNorm = (1.0f-_alpha) * 1.0f/edgeEnergyGradientNorm;

		float32 gradSize = 0.0f;
		for( unsigned i = 0; i < gradient.Size(); ++i ) {
			gradient[i] = regionEnergyGradientNorm * regionEnergyGradient[i] + edgeEnergyGradientNorm * edgeEnergyGradient[i];
			gradSize += gradient[i] * gradient[i];
		}
		return sqrt( gradSize );
	}

	void
	SetRegionStat( const RegionType1 &region )
		{ _region1 = region; }
	void
	SetRegionEdge( const RegionType2 &region )
		{ _region2 = region; }

	float32
	GetAlpha() const
		{ return _alpha; }

	void
	SetAlpha( float32 a )
		{ _alpha = a; }
	
private:
	float32
	ComputeValueAtPointRegion( const PointCoordinate &pos )
	{
		int value = _region1.GetElementWorldCoords( pos );
		
		return -(this->LogProbabilityRatio( value, pos ));
	}

	float32
	ComputeValueAtPointEdge( const PointCoordinate &pos )
	{
		return _region2.GetElementWorldCoords( pos );
	}

	void
	FillSampleValuesBuffers( const SampleSet & samples )
	{
		int32 sampleCount = samples.Size();
		_regionValBuffer.resize( sampleCount );
		_edgeValBuffer.resize( sampleCount );
		for( int32 i = 0; i < sampleCount; ++i ) {
			_regionValBuffer[ i ] = ComputeValueAtPointRegion( samples[i] );
			_edgeValBuffer[ i ] = ComputeValueAtPointEdge( samples[i] );
		}
	}

	PointCoordinate
	ComputePointGradient( unsigned k, ContourType &curve, ValuesAtSamplesBuffer &valBuffer )
	{
		PointCoordinate gradient = PointCoordinate( 0.0f );

		for( int32 i = k - Degree; i <= (int32)(k + Degree); ++i ) {
			gradient += curve.GetPointCyclic( i ) * ComputeIntegral( k, i, curve, valBuffer );
		}
		gradient = VectorDimensionsShiftRight( gradient );
		gradient[0] *= -1;

		return gradient;
	}

	float32
	ComputeIntegral( int32 k, int32 i, ContourType &curve, ValuesAtSamplesBuffer &valBuffer )
	{
		float32 result = 0.0f;
		int32 L = max( k, i ) - Degree;
		int32 U = min( k, i ) + 1;
		int32 sampleCount = curve.GetSamplePoints().Size();

		for( int32 j = L*_sampleFrequency; j < U*_sampleFrequency; ++j ) {

			int32 idx = MOD( j, sampleCount );

			result += valBuffer[ idx ] * Qki( k, i, j );
		}
		return result / (float32)_sampleFrequency;
	}
	
	float32
	Qki( int32 k, int32 i, int32 tR )
	{
		//int32 tLow = tR / _sampleFrequency;
		int32 tLow = floor( (float)tR / (float)_sampleFrequency );
		int32 nk = k - tLow;
		int32 ni = i - tLow;
		int32 t = tR - tLow * _sampleFrequency;
		
		if( nk > (int32)Degree ) { return 0.0f; }
		if( nk < 0 ) { return 0.0f; }
		if( ni > (int32)Degree ) { return 0.0f; }
		if( ni < 0 ) { return 0.0f; }
		if( t > (int32)_sampleFrequency ) { return 0.0f; }
		if( t < 0 ) { return 0.0f; }

		return Q[nk][ni][ t ];
	}

	void
	RecalculateQki( ContourType &curve )
	{
		const BFValVector &values = curve.GetLastBasisFunctionValues();
		const BFValVector &derivValues = curve.GetLastBasisFunctionDerivationValues();
		_sampleFrequency = curve.GetLastSampleFrequency();

		Q.resize( Degree+1 );
		for( int i = 0; i <= (int)Degree; ++i ) {
			Q[i].resize( Degree+1 );
			for( int j = 0; j <= (int)Degree; ++j ) {
				Q[i][j].resize( _sampleFrequency );
				for( int k = 0; k < _sampleFrequency; ++k ) {
					Q[i][j][k] = values[k][i] * derivValues[k][j];
				}
			}
		}
	}

	float32 		_alpha;

	RegionType1		_region1;
	RegionType2		_region2;
	ValuesAtSamplesBuffer	_edgeValBuffer;
	ValuesAtSamplesBuffer	_regionValBuffer;

	std::vector< std::vector< std::vector< typename ContourType::Type > > > Q;
	
	int32 _sampleFrequency;
};


template< typename ContourType, typename RegionType, typename Distribution >
class RegionImageEnergy : public Distribution
{
public:
	typedef  M4D::Imaging::Geometry::PointSet< typename ContourType::PointType > 	GradientType;
	typedef typename ContourType::PointType	PointCoordinate;
	typedef typename ContourType::BFValVector	BFValVector;
	typedef typename RegionType::ElementType	ElementType;
	typedef std::vector< float32 >			ValuesAtSamplesBuffer;
	typedef typename ContourType::SamplePointSet	SampleSet;
	static const unsigned Degree = ContourType::Degree;

	RegionImageEnergy():_sampleFrequency( -1 )
		{}

	void
	ResetEnergy() {}

	float32
	GetParametersGradient( ContourType &curve, GradientType &gradient )
	{
		if( curve.Size() != gradient.Size() ) {
			//TODO - solve problem
		}

		if( _sampleFrequency != (int32)curve.GetLastSampleFrequency() ) {
			RecalculateQki( curve );
		}

		FillSampleValuesBuffer( curve.GetSamplePoints() );		

		float32 gradSize = 0.0f;
		for( unsigned i = 0; i < gradient.Size(); ++i ) {
			gradient[i] = ComputePointGradient( i, curve );
			gradSize += gradient[i] * gradient[i];
		}
		return sqrt( gradSize );
	}

	void
	SetRegionStat( const RegionType &region )
		{ _region = region; }

private:
	float32
	ComputeValueAtPoint( const PointCoordinate &pos )
	{
		int value = _region.GetElementWorldCoords( pos );

		float32 val1 = this->LogProbabilityRatio( value, pos );
		
		return -val1;
	}

	void
	FillSampleValuesBuffer( const SampleSet & samples )
	{
		int32 sampleCount = samples.Size();
		_valBuffer.resize( sampleCount );
		for( int32 i = 0; i < sampleCount; ++i ) {
			//TODO interpolation
			/*float32 x = samples[i][0];
			float32 y = samples[i][1];
			RasterPos pos = RasterPos( ROUND( x ), ROUND( y ) );*/
			_valBuffer[ i ] = ComputeValueAtPoint( samples[i] );
		}
	}

	PointCoordinate
	ComputePointGradient( unsigned k, ContourType &curve )
	{
		PointCoordinate gradient = PointCoordinate( 0.0f );

		for( int32 i = k - Degree; i <= (int32)(k + Degree); ++i ) {
			gradient += curve.GetPointCyclic( i ) * ComputeIntegral( k, i, curve );
		}
		gradient = VectorDimensionsShiftRight( gradient );
		gradient[0] *= -1;

		return gradient;
	}

	float32
	ComputeIntegral( int32 k, int32 i, ContourType &curve )
	{
		float32 result = 0.0f;
		int32 L = max( k, i ) - Degree;
		int32 U = min( k, i ) + 1;
		int32 sampleCount = curve.GetSamplePoints().Size();


		for( int32 j = L*_sampleFrequency; j < U*_sampleFrequency; ++j ) {

			int32 idx = MOD( j, sampleCount );

			result += _valBuffer[ idx ] * Qki( k, i, j );
		}
		return result / (float32)_sampleFrequency;
	}
	
	float32
	Qki( int32 k, int32 i, int32 tR )
	{
		//int32 tLow = tR / _sampleFrequency;
		int32 tLow = static_cast< int32 >( floor( (float)tR / (float)_sampleFrequency ) );
		int32 nk = k - tLow;
		int32 ni = i - tLow;
		int32 t = tR - tLow * _sampleFrequency;
		
		if( nk > (int32)Degree ) { return 0.0f; }
		if( nk < 0 ) { return 0.0f; }
		if( ni > (int32)Degree ) { return 0.0f; }
		if( ni < 0 ) { return 0.0f; }
		if( t > (int32)_sampleFrequency ) { return 0.0f; }
		if( t < 0 ) { return 0.0f; }

		return Q[nk][ni][ t ];
	}

	void
	RecalculateQki( ContourType &curve )
	{
		const BFValVector &values = curve.GetLastBasisFunctionValues();
		const BFValVector &derivValues = curve.GetLastBasisFunctionDerivationValues();
		_sampleFrequency = curve.GetLastSampleFrequency();
		
		ASSERT( derivValues.size() == static_cast<unsigned>(_sampleFrequency) );
		ASSERT( values.size() == static_cast<unsigned>(_sampleFrequency) );

		Q.resize( Degree+1 );
		for( int i = 0; i <= (int)Degree; ++i ) {
			Q[i].resize( Degree+1 );
			for( int j = 0; j <= (int)Degree; ++j ) {
				Q[i][j].resize( _sampleFrequency );
				for( int k = 0; k < _sampleFrequency; ++k ) {
					Q[i][j][k] = values[k][i] * derivValues[k][j];
				}
			}
		}
	}

	RegionType		_region;
	ValuesAtSamplesBuffer	_valBuffer;

	std::vector< std::vector< std::vector< typename ContourType::Type > > > Q;
	
	int32 _sampleFrequency;
};

template< typename ContourType >
class InternalCurveEnergy
{
public:
	typedef  M4D::Imaging::Geometry::PointSet< typename ContourType::PointType > 	GradientType;
	typedef typename ContourType::PointType	PointCoordinate;
	typedef typename ContourType::BFValVector	BFValVector;
	static const int32 Degree = ContourType::Degree;
	static const int32 TableRowSize = 2*Degree - 1;

	InternalCurveEnergy(): _gamma( 1.0f )
		{
			PrepareTables();
		}

	void
	ResetEnergy() {}

	float32
	GetGamma()const
		{ return _gamma; }

	void
	SetGamma( float32 val )
		{ _gamma = val; }

	float32
	GetParametersGradient( ContourType &curve, GradientType &gradient )
	{
		if( curve.Size() != gradient.Size() ) {
			//TODO - solve problem
		}
		
		ComputeC( curve );
		
		_differences.Resize( gradient.Size() );
		for( int32 i = 0; i < (int32)_differences.Size(); ++i ) {
			_differences[i] = curve.GetPointCyclic( i - 1 ) - curve.GetPointCyclic( i );
		}
		float32 gradSize = 0.0f;
		for( unsigned i = 0; i < gradient.Size(); ++i ) {
			gradient[i] = ComputePointGradient( i, curve );
			gradSize += gradient[i] * gradient[i];
		}
		return sqrt( gradSize );
	}
private:
	PointCoordinate
	ComputePointGradient( unsigned k, ContourType &curve )
	{
		PointCoordinate gradient = PointCoordinate( 0.0f );

		for( int32 l = -Degree + 1; l < (int32)(Degree); ++l ) {
			//const PointCoordinate &pl = curve.GetPointCyclic( k + l );
			const PointCoordinate &pl = _differences.GetPointCyclic( k + l );
			for( int32 m = -Degree + 1; m < (int32)(Degree); ++m ) {
				//const PointCoordinate &pm = curve.GetPointCyclic( k + m );
				const PointCoordinate &pm = _differences.GetPointCyclic( k + m );
				for( int32 n = -Degree + 1; n < (int32)(Degree); ++n ) {
					//const PointCoordinate &pn = curve.GetPointCyclic( k + n );
					const PointCoordinate &pn = _differences.GetPointCyclic( k + n );
					for( unsigned i = 0; i < 2; ++i ) {
						gradient[i] += 4 * pl[i] * pm[i] * pn[i] * H1(l,m,n);
						gradient[i] += 4 * pl[i] * pm[(i+1) % 2] * pn[(i+1) % 2] * H1(l,m,n);
					}
				}
			}
			for( unsigned i = 0; i < 2; ++i ) {
				gradient[i] -= 4 * c * pl[i] * H2(l);
			}
		}
		return gradient;
	}

	float32&
	H1( int32 l, int32 m, int32 n )
	{
		int32 lIdx = l + Degree - 1;
		int32 mIdx = m + Degree - 1;
		int32 nIdx = n + Degree - 1;
		return h1Table[ lIdx * TableRowSize*TableRowSize + mIdx*TableRowSize + nIdx ];
	}
	
	float32&
	H2( int32 l )
	{
		return h2Table[ l + Degree - 1];
	}

	void
	ComputeC( ContourType &curve )
	{ 
		c = _gamma * Sqr( BSplineLength( curve ) / curve.GetSegmentCount() ); 
	}

	void
	PrepareTables()
	{
		static const int32 SampleFrequency = 10;
		typename ContourType::BFValVector basisFunctionValues;
		basisFunctionValues.resize( SampleFrequency );
		float32 t = 0.0f;
		float32 dt = 1.0f / SampleFrequency;
		for( int32 i=0; i < SampleFrequency; ++i, t += dt ) {
			ContourType::CurveBasis::DerivationsAtPoint( t, basisFunctionValues[ i ] );	
		}

		for( int32 l = -Degree + 1; l < (int32)(Degree); ++l ) {
			for( int32 m = -Degree + 1; m < (int32)(Degree); ++m ) {
				for( int32 n = -Degree + 1; n < (int32)(Degree); ++n ) {
					float32 h1Result = 0.0f;
					
					if( l >= -2 && l <= 3 && m >= -2 && m <= 3 && n >= -2 && n <= 3 ) {
						int32 tmpBoundUp = min( 1, - max( l, m, n ) + 1 );
						int32 tmpBoundDown = max( -3, - min( l, m, n ) -3 );
						for( int32 s = tmpBoundDown * SampleFrequency ; s < tmpBoundUp * SampleFrequency; ++s ) {
							int32 pomt = MOD(s, SampleFrequency );
							int32 pomDt = s / SampleFrequency;
							if( SampleFrequency * pomDt > s ){ --pomDt; }
							h1Result += 
								basisFunctionValues[pomt][  -pomDt] * 
								basisFunctionValues[pomt][-l-pomDt] * 
								basisFunctionValues[pomt][-m-pomDt] * 
								basisFunctionValues[pomt][-n-pomDt];
						}
					}

					H1( l, m, n ) = h1Result;
				}
			}
			float32 h2Result = 0.0f;
			if( l >= -2 && l <= 3 ) {
				int32 tmpBoundUp = min( 1, -l + 1 );
				int32 tmpBoundDown = max( -3, -l -3 );
				for( int32 s = tmpBoundDown * SampleFrequency ; s < tmpBoundUp * SampleFrequency; ++s ) {
					int32 pomt = MOD(s, SampleFrequency );
					int32 pomDt = s / SampleFrequency;
					if( SampleFrequency * pomDt > s ){ --pomDt; }
					h2Result += basisFunctionValues[pomt][-pomDt] * basisFunctionValues[pomt][-l-pomDt];
				}
			}
			H2( l ) = h2Result;
		}

	}

private:
	GradientType _differences;

	float32 _gamma;


	float32 c;

	float32		h1Table[TableRowSize*TableRowSize*TableRowSize];
	float32 	h2Table[TableRowSize];

};

/**
 * Energy functional doing nothing :-) testing purpose. 
 **/
class DummyEnergy1
{
public:
	template< typename ContourType >
	float32
	GetParametersGradient( ContourType &curve, 
			M4D::Imaging::Geometry::PointSet< typename ContourType::PointType > &gradient )
	{
		return 0.0f;
	}

	void
	ResetEnergy() {}
};

/**
 * Energy functional doing nothing :-) testing purpose. 
 **/
class DummyEnergy2
{
public:
	template< typename ContourType >
	float32
	GetParametersGradient( ContourType &curve, 
			M4D::Imaging::Geometry::PointSet< typename ContourType::PointType > &gradient )
	{
		return 0.0f;
	}

	void
	ResetEnergy() {}
};

/**
 * Energy functional doing nothing :-) testing purpose. 
 **/
class DummyEnergy3
{
public:
	template< typename ContourType >
	float32
	GetParametersGradient( ContourType &curve, 
			M4D::Imaging::Geometry::PointSet< typename ContourType::PointType > &gradient )
	{
		return 0.0f;
	}

	void
	ResetEnergy() {}
};


}/*namespace Algorithms*/
}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#include "Imaging/EnergyModels2.h"

#endif /*ENERGY_MODELS_H*/
