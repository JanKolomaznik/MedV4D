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
CalculateHistograms( MultiHistogram< uint32, 2 >& jointHist,
		 Histogram< uint32 >& inputHist,
		 Histogram< uint32 >& refHist,
		 Image< ElementType, 3 >& inputImage,
		 Image< ElementType, 3 >& refImage )
{
	jointHist.Reset();
	inputHist.Reset();
	refHist.Reset();
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
				inputHist.IncCell( *in );
				refHist.IncCell( *ref );
				in += inXStride;
				ref += refXStride;
			}
		}
	std::cout << "Done" << std::endl;
}

template< typename ElementType, uint32 dim >
ImageRegistration< ElementType, dim >
::ImageRegistration( typename ImageRegistration< ElementType, dim >::Properties  * prop )
	: PredecessorType( prop ),
	  jointHistogram( std::vector< int32 >( 2, TypeTraits< ElementType >::Min ), std::vector< int32 >( 2, TypeTraits< ElementType >::Max ) ),
	  inputImageHistogram( TypeTraits< ElementType >::Min, TypeTraits< ElementType >::Max ),
	  referenceImageHistogram( TypeTraits< ElementType >::Min, TypeTraits< ElementType >::Max )
{
	this->_name = "ImageRegistration";
}

template< typename ElementType, uint32 dim >
ImageRegistration< ElementType, dim >
::ImageRegistration()
	: PredecessorType( new Properties() ),
	  jointHistogram( std::vector< int32 >( 2, TypeTraits< ElementType >::Min ), std::vector< int32 >( 2, TypeTraits< ElementType >::Max ) ),
	  inputImageHistogram( TypeTraits< ElementType >::Min, TypeTraits< ElementType >::Max ),
	  referenceImageHistogram( TypeTraits< ElementType >::Min, TypeTraits< ElementType >::Max )
{
	this->_name = "ImageRegistration";
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
	bool result = false;
	result = this->ExecuteTransformation();
	//if ( referenceImage ) CalculateHistograms< ElementType > ( jointHistogram, inputImageHistogram, referenceImageHistogram, *(this->out), *(referenceImage) );
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
	
	/*if ( referenceImage && this->in )
	{

		Vector< uint32, 3 > size;
 		Vector< int32, 3 > strides;
		this->in->GetPointer( size, strides );
		uint32 inWidth = size[0];
		uint32 inHeight = size[1];
		uint32 inDepth = size[2];
		referenceImage->GetPointer( size, strides );
		uint32 refWidth = size[0];
		uint32 refHeight = size[1];
		uint32 refDepth = size[2];

		typedef typename PredecessorType::CoordType::CoordinateType		CoordType;

		this->SetSampling( CreateVector< CoordType >( (CoordType)refWidth/(CoordType)inWidth, (CoordType)refHeight/(CoordType)inHeight, (CoordType)refDepth/(CoordType)inDepth ) );

		this->_callPrepareOutputDatasets = true;

	}*/
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

