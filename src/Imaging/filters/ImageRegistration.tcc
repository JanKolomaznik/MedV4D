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
	for ( j = 0; j < refHeight; ++j )
	{
		ElementType *in = sin + j*inYStride;
		ElementType *ref = sref + j*refYStride;
		for ( i = 0; i < refWidth; ++i )
		{
			index[0] = *in;
			index[1] = *ref;
			jointHist.IncCell( index );
			in += inXStride;
			ref += refXStride;
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
	for ( k = 0; k < refDepth; ++k )
		for ( j = 0; j < refHeight; ++j )
		{
			ElementType *in = sin + k*inZStride + j*inYStride;
			ElementType *ref = sref + k*refZStride + j*refYStride;
			for ( i = 0; i < refWidth; ++i )
			{
				index[0] = *in;
				index[1] = *ref;
				jointHist.IncCell( index );
				in += inXStride;
				ref += refXStride;
			}
		}
}

template< typename ElementType, uint32 dim >
ImageRegistration< ElementType, dim >
::ImageRegistration( typename ImageRegistration< ElementType, dim >::Properties  * prop )
	: PredecessorType( prop ),
	  jointHistogram( std::vector< int32 >( 2, HISTOGRAM_MIN_VALUE ), std::vector< int32 >( 2, HISTOGRAM_MAX_VALUE ) ),
	  _criterion( new NormalizedMutualInformation< HistCellType >() ),
	  _optimization( new PowellOptimization< ElementType, double, 3 * dim >() )
{
	this->_name = "ImageRegistration";
}

template< typename ElementType, uint32 dim >
ImageRegistration< ElementType, dim >
::ImageRegistration()
	: PredecessorType( new Properties() ),
	  jointHistogram( std::vector< int32 >( 2, HISTOGRAM_MIN_VALUE ), std::vector< int32 >( 2, HISTOGRAM_MAX_VALUE ) ),
	  _criterion( new NormalizedMutualInformation< HistCellType >() ),
	  _optimization( new PowellOptimization< ElementType, double, 3 * dim >() )
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
double
ImageRegistration< ElementType, dim >
::OptimizationFunction( Vector< double, 3 * dim >& v )
{
	Vector< double, dim > v1, v2, v3;
	for ( uint32 i = 0; i < dim; ++i )
	{
		v1[i] = v[i];
		v2[i] = v[dim + i];
		v3[i] = v[2 * dim + i];
	}
	this->SetRotation( v1 );
	this->SetScale( v2 );
	this->SetTranslation( v3 );
	this->ExecuteTransformation();
	double res = 1.0;
	if ( referenceImage )
	{
		CalculateHistograms< ElementType > ( jointHistogram, *(this->out), *(referenceImage) );
		uint32 size = 1;
		Vector< uint32, dim > sizeVector;
		Vector< int32, dim > strideVector;
		referenceImage->GetPointer( sizeVector, strideVector );
		for ( uint32 i = 0; i < dim; ++i ) size *= sizeVector[i];
		res = _criterion->compute( jointHistogram, size );
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
	Vector< double, 3 * dim > v;
	for ( uint32 i = 0; i < dim; ++i )
	{
		v[i] = 0;
		v[dim + i] = 1;
		v[2 * dim + i] = 0;
	}
	double fret;
	_optimization->optimize( v, fret, this );
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

