#ifndef _IMAGE_CONVERTOR_H
#error File ImageConvertor.tcc cannot be included directly!
#else

namespace M4D
{

namespace Imaging
{

template< typename OutputImageType, typename Convertor = DefaultConvertor >
ImageConvertor< OutputImageType >
::ImageConvertor( ImageConvertor< OutputImageType >::Properties  * prop )
	: PredecessorType( prop )
{

}

template< typename OutputImageType, typename Convertor = DefaultConvertor >
ImageConvertor
::ImageConvertor()
	: PredecessorType( new Properties() )
{
	
}

template< typename OutputImageType, typename Convertor = DefaultConvertor >
bool
ImageConvertor
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{

}

template< typename OutputImageType, typename Convertor = DefaultConvertor >
void
ImageConvertor
::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();
}

template< typename OutputImageType, typename Convertor = DefaultConvertor >
void
ImageConvertor
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );
}

template< typename OutputImageType, typename Convertor = DefaultConvertor >
void
ImageConvertor
::AfterComputation( bool successful )
{
	PredecessorType::AfterComputation( successful );
}


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_IMAGE_CONVERTOR_H*/
