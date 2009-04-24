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

/*template< typename ElementType >
void
CalculateHistograms( MultiHistogram< uint32, 2 >& jointHist,
		 Histogram< uint32 >& inputHist,
		 Histogram< uint32 >& refHist,
		 Image< ElementType, 3 >& inputImage,
		 Image< ElementType, 3 >& refImage )
{
	jointHistogram.Reset();
	inputImageHistogram.Reset();
	referenceImageHistogram.Reset();
	std::vector< ElementType > index(2);
	uint32 i, j, k;
	for ( i = 
	
}*/

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
::SetReferenceImage( typename ImageType::Ptr ref )
{
	referenceImage = ref;
}

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*IMAGE_REGISTRATION_H_*/

/** @} */

