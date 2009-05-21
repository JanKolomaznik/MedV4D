/**
 * @ingroup imaging 
 * @author Szabolcs Grof
 * @file ImageRegistration.tcc 
 * @{ 
 **/

#ifndef IMAGE_REGISTRATION_H_
#error File ImageRegistration.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

template< typename ElementType >
void
CalculateHistograms( MultiHistogram< typename ImageRegistration< ElementType, 2 >::HistCellType, 2 >& jointHist,
		 Image< ElementType, 2 >& inputImage,
		 Image< ElementType, 2 >& refImage )
{
	jointHist.Reset();
	std::vector< int32 > index(2);
	uint32 i, j;
	ElementType *sin, *sref;
	Vector< uint32, 2 > size;
        Vector< int32, 2 > strides;
	sin = inputImage.GetPointer( size, strides );
	int32 inXStride = strides[0];
	int32 inYStride = strides[1];
	sref = refImage.GetPointer( size, strides );
	int32 refXStride = strides[0];
	int32 refYStride = strides[1];
	uint32 refWidth = size[0];
	uint32 refHeight = size[1];
	typename ImageRegistration< ElementType, 2 >::HistCellType base = 1.0 / ((refHeight/HISTOGRAM_SAMPLING_DIVISOR) * (refWidth/HISTOGRAM_SAMPLING_DIVISOR));
	for ( j = 0; j < refHeight / HISTOGRAM_SAMPLING_DIVISOR; ++j )
	{
		ElementType *in = sin + HISTOGRAM_SAMPLING_DIVISOR*j*inYStride;
		ElementType *ref = sref + HISTOGRAM_SAMPLING_DIVISOR*j*refYStride;
		for ( i = 0; i < refWidth / HISTOGRAM_SAMPLING_DIVISOR; ++i )
		{
			index[0] = (*in) / HISTOGRAM_VALUE_DIVISOR;
			index[1] = (*ref) / HISTOGRAM_VALUE_DIVISOR;
			jointHist.SetValueCell( index, jointHist.Get( index ) + base );
			in += HISTOGRAM_SAMPLING_DIVISOR * inXStride;
			ref += HISTOGRAM_SAMPLING_DIVISOR * refXStride;
		}
	}
}

template< typename ElementType >
void
CalculateHistograms( MultiHistogram< typename ImageRegistration< ElementType, 3 >::HistCellType, 2 >& jointHist,
		 Image< ElementType, 3 >& inputImage,
		 Image< ElementType, 3 >& refImage )
{
	jointHist.Reset();
	std::vector< int32 > index(2);
	uint32 i, j, k;
	ElementType *sin, *sref;
	Vector< uint32, 3 > size;
        Vector< int32, 3 > strides;
	sin = inputImage.GetPointer( size, strides );
	int32 inXStride = strides[0];
	int32 inYStride = strides[1];
	int32 inZStride = strides[2];
	sref = refImage.GetPointer( size, strides );
	int32 refXStride = strides[0];
	int32 refYStride = strides[1];
	int32 refZStride = strides[2];
	uint32 refWidth = size[0];
	uint32 refHeight = size[1];
	uint32 refDepth = size[2];
	typename ImageRegistration< ElementType, 3 >::HistCellType base = 1.0 / ((refHeight/HISTOGRAM_SAMPLING_DIVISOR) * (refWidth/HISTOGRAM_SAMPLING_DIVISOR) * refDepth);
	for ( k = 0; k < refDepth; ++k )
		for ( j = 0; j < refHeight / HISTOGRAM_SAMPLING_DIVISOR; ++j )
		{
			ElementType *in = sin + k*inZStride + HISTOGRAM_SAMPLING_DIVISOR*j*inYStride;
			ElementType *ref = sref + k*refZStride + HISTOGRAM_SAMPLING_DIVISOR*j*refYStride;
			for ( i = 0; i < refWidth / HISTOGRAM_SAMPLING_DIVISOR; ++i )
			{
				index[0] = (*in) / HISTOGRAM_VALUE_DIVISOR;
				index[1] = (*ref) / HISTOGRAM_VALUE_DIVISOR;
				jointHist.SetValueCell( index, jointHist.Get( index ) + base );
				in += HISTOGRAM_SAMPLING_DIVISOR * inXStride;
				ref += HISTOGRAM_SAMPLING_DIVISOR * refXStride;
			}
		}
}

template< typename ElementType, uint32 dim >
ImageRegistration< ElementType, dim >
::ImageRegistration( typename ImageRegistration< ElementType, dim >::Properties  * prop )
	: PredecessorType( prop ),
	  jointHistogram( std::vector< int32 >( 2, HISTOGRAM_MIN_VALUE ), std::vector< int32 >( 2, HISTOGRAM_MAX_VALUE ) ),
	  _criterion( new NormalizedMutualInformation< HistCellType >() ),
	  _optimization( new PowellOptimization< ElementType, double, 2 * dim >() ),
	  _automatic( false )
{
	this->_name = "ImageRegistration";
}

template< typename ElementType, uint32 dim >
ImageRegistration< ElementType, dim >
::ImageRegistration()
	: PredecessorType( new Properties() ),
	  jointHistogram( std::vector< int32 >( 2, HISTOGRAM_MIN_VALUE ), std::vector< int32 >( 2, HISTOGRAM_MAX_VALUE ) ),
	  _criterion( new NormalizedMutualInformation< HistCellType >() ),
	  _optimization( new PowellOptimization< ElementType, double, 2 * dim >() ),
	  _automatic( false )
{
	this->_name = "ImageRegistration";
}

template< typename ElementType, uint32 dim >
ImageRegistration< ElementType, dim >
::~ImageRegistration()
{
	delete _criterion;
	delete _optimization;
}

template< typename ElementType, uint32 dim >
void
ImageRegistration< ElementType, dim >
::SetAutomaticMode( bool mode )
{
	_automatic = mode;
}

template< typename ElementType, uint32 dim >
double
ImageRegistration< ElementType, dim >
::OptimizationFunction( Vector< double, 2 * dim >& v )
{
	Vector< double, dim > v1, v2;
	for ( uint32 i = 0; i < dim; ++i )
	{
		v1[i] = v[i];
		v2[i] = v[dim + i];
	}
	this->SetRotation( v1 );
	this->SetTranslation( v2 );
	this->ExecuteTransformation();
	double res = 1.0;
	if ( referenceImage )
	{
		CalculateHistograms< ElementType > ( jointHistogram, *(this->out), *(referenceImage) );
		res = _criterion->compute( jointHistogram );
		std::cout << res << std:: endl;
	}
	return 1.0 / res;
}

template< typename ElementType, uint32 dim >
bool
ImageRegistration< ElementType, dim >
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
	utype = utype;
	if ( !( this->_readerBBox->WaitWhileDirty() == MS_MODIFIED ) ) {
		this->_writerBBox->SetState( MS_CANCELED );
		return false;
	}
	if ( _automatic )
	{
		Vector< double, 2 * dim > v;
		for ( uint32 i = 0; i < dim; ++i )
		{
			v[i] = 0;
			v[dim + i] = 0;
		}
		double fret;
		_optimization->optimize( v, fret, this );
	}
	else this->ExecuteTransformation();
	bool result = true;
	if( result ) {
		this->_writerBBox->SetModified();
	} else {
		this->_writerBBox->SetState( MS_CANCELED );
	}
	return result;
}

template< typename ElementType, uint32 dim >
void
ImageRegistration< ElementType, dim >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{

        PredecessorType::BeforeComputation( utype );
	
	if ( referenceImage && this->in )
	{

		Vector< uint32, dim > inSize;
		Vector< uint32, dim > refSize;
 		Vector< int32, dim > strides;
		this->in->GetPointer( inSize, strides );
		referenceImage->GetPointer( refSize, strides );

		typedef typename PredecessorType::CoordType::CoordinateType		CoordType;

		Vector< CoordType, dim > scale;
		for ( uint32 i = 0; i < dim; ++i ) scale[i] = static_cast< CoordType >( inSize[i] * this->in->GetDimensionExtents( i ).elementExtent )/static_cast< CoordType >( refSize[i] * referenceImage->GetDimensionExtents( i ).elementExtent );
		this->SetScale( scale );

		Vector< CoordType, dim > sampling;

		for ( uint32 i = 0; i < dim; ++i ) sampling[i] = static_cast< CoordType >( refSize[i] )/static_cast< CoordType >( inSize[i] );

		this->SetSampling( sampling );

		this->_callPrepareOutputDatasets = true;

	}
}

template< typename ElementType, uint32 dim >
void
ImageRegistration< ElementType, dim >
::SetReferenceImage( typename ImageType::Ptr ref )
{
	referenceImage = ref;
}

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*IMAGE_REGISTRATION_H_*/

/** @} */

