/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageConnection.tcc 
 * @{ 
 **/

#ifndef _IMAGE_CONNECTION_H
#error File ImageConnection.tcc cannot be included directly!
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
CallImageFactoryRealloc( 
			Image< ElementType, 2 > &image,  
			int32 		minimums[ 2 ], 
			int32 		maximums[ 2 ], 
			float32		elementExtents[ 2 ]
	    )
{
	ImageFactory::ReallocateImage2DData< ElementType >( 
		image, 
		maximums[0]-minimums[0], 
		maximums[1]-minimums[1], 
		elementExtents[0], 
		elementExtents[1]
	);
}

template< typename ElementType >
void 
CallImageFactoryRealloc( 
			Image< ElementType, 3 > &image,  
			int32 		minimums[ 3 ], 
			int32 		maximums[ 3 ], 
			float32		elementExtents[ 3 ]
	    )
{
	ImageFactory::ReallocateImage3DData< ElementType >( 
		image, 
		maximums[0]-minimums[0], 
		maximums[1]-minimums[1], 
		maximums[2]-minimums[2], 
		elementExtents[0], 
		elementExtents[1],
		elementExtents[2]
	);
}



//*****************************************************************************
template< typename ElementType, unsigned dimension >
ImageConnection< Image< ElementType, dimension > >
::ImageConnection( bool ownsDataset )
{
	if( ownsDataset ) {
		this->_image = typename M4D::Imaging::Image< ElementType, dimension >::Ptr( 
			new M4D::Imaging::Image< ElementType, dimension >() 
			);
	}
}

template< typename ElementType, unsigned dimension >
void
ImageConnection< Image< ElementType, dimension > >
::ConnectProducer( OutputPort& outputPort )
{
	OutputImagePort *port = 
		dynamic_cast< OutputImagePort * >( &outputPort );
	if( port ) {
			port->Plug( *this );
			_producer = port;
	} else {
		throw ConnectionInterface::EMismatchPortType();
	}
}

template< typename ElementType, unsigned dimension >
void
ImageConnection< Image< ElementType, dimension > >
::ConnectConsumer( InputPort& inputPort )
{
	//We allow only ports of exact type or generic port - nothing more!!!
	if( typeid(inputPort) == typeid( InputImagePort &) ||
		typeid(inputPort) == typeid( InputPortAbstractImage &)
	  )
	{
		this->PushConsumer( inputPort );
	} else {
		throw ConnectionInterface::EMismatchPortType();
	}
}

template< typename ElementType, unsigned dimension >
void
ImageConnection< Image< ElementType, dimension > >
::PutImage( typename M4D::Imaging::Image< ElementType, dimension >::Ptr image )
{
	if( !image ) {
		throw AbstractImageConnectionInterface::EInvalidImage();
	}
	this->_image = image;

	RouteMessage( 
			MsgDatasetPut::CreateMsg(), 
			PipelineMessage::MSS_NORMAL,
			FD_BOTH	
		);
}

template< typename ElementType, unsigned dimension >
void
ImageConnection< Image< ElementType, dimension > >
::PutImage( M4D::Imaging::AbstractImage::AImagePtr image )
{
	typename M4D::Imaging::Image< ElementType, dimension >::Ptr typedImage = 
		M4D::Imaging::Image< ElementType, dimension >::CastAbstractImage( image );

	PutImage( typedImage );
}

template< typename ElementType, unsigned dimension >
void
ImageConnection< M4D::Imaging::Image< ElementType, dimension > >
::SetImageSize( 
		uint32		dim,
		int32 		minimums[], 
		int32 		maximums[], 
		float32		elementExtents[]
	    )
{
	if( !_image ) {
		throw ENoImageAssociated();
	}
	if( dim != dimension ) {
		throw ErrorHandling::EBadDimension();
	}

	//TODO - check if locking should be done here
	_image->UpgradeToExclusiveLock();

	CallImageFactoryRealloc( *_image, minimums, maximums, elementExtents );

	_image->DowngradeFromExclusiveLock();
}





} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_IMAGE_CONNECTION_H*/


/** @} */

