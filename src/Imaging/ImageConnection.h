#ifndef _IMAGE_CONNECTION_H
#define _IMAGE_CONNECTION_H

#include "Imaging/ConnectionInterface.h"
#include "Imaging/Image.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/AbstractFilter.h"
#include "Imaging/ImagePorts.h"

namespace M4D
{
namespace Imaging
{


/**
 * Not supposed to instantiate - use only as substitution for typed images connections.
 **/
class AbstractImageConnection : public ConnectionInterface
{
public:

	virtual const AbstractImage &
	GetAbstractImageReadOnly()const = 0;

	virtual AbstractImage &
	GetAbstractImage()const = 0;

	const AbstractDataSet &
	GetDatasetReadOnly()const
		{ 
			return GetAbstractImageReadOnly();
		}
	
	AbstractDataSet &
	GetDataset()const
		{
			return GetAbstractImage();
		}

protected:

};

//We prohibit general usage - only specialized templates used.
template< typename ImageTemplate >
class ImageConnection;


template< typename ElementType, unsigned dimension >
class ImageConnection< Image< ElementType, dimension > >
	: public AbstractImageConnection
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > Image;
	typedef typename M4D::Imaging::InputPortImageFilter< Image > InputImagePort;
	typedef typename M4D::Imaging::OutputPortImageFilter< Image > OutputImagePort;
	
	ImageConnection( bool ownsDataset ); 
	~ImageConnection() {}

	
	void
	ConnectConsumer( InputPort& inputPort );

	void
	ConnectProducer( OutputPort& outputPort );
	

		void
	PutImage( typename M4D::Imaging::Image< ElementType, dimension >::Ptr image );

	void
	PutImage( M4D::Imaging::AbstractImage::AImagePtr image );


	Image &
	GetImage()const 
		{ if( !_image ) { throw ENoImageAssociated(); }
			return *_image;
		}

	const Image &
	GetImageReadOnly()const
		{ if( !_image ) { throw ENoImageAssociated(); }
			return *_image;
		}

	AbstractDataSet &
	GetDataset()const
		{ 
			return GetImage();
		}

	const AbstractDataSet &
	GetDatasetReadOnly()const
		{ 
			return GetImageReadOnly();
		}

	const AbstractImage &
	GetAbstractImageReadOnly()const
		{
			return GetImageReadOnly();
		}

	AbstractImage &
	GetAbstractImage()const
		{
			return GetImage();
		}

	void
	SetImageSize( 
			size_t 		minimums[ dimension ], 
			size_t 		maximums[ dimension ], 
			float32		elementExtents[ dimension ]
		    );

	void
	RouteMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle,
		FlowDirection				direction
		);

protected:

	/**
	 * Hidden default constructor - we don't allow direct
	 * construction of object of this class.
	 **/
	ImageConnection() {};
	
	ImageConnection( typename Image::Ptr image ) 
		: _image( image ) {}

	typename Image::Ptr			_image;
	OutputImagePort				*_input;
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( ImageConnection );

public:
	/**
	 * Exception thrown when requiring image object and none 
	 * is available.
	 **/
	class ENoImageAssociated
	{
		//TODO
	};
	class EInvalidImage
	{
		//TODO
	};
};



}/*namespace Imaging*/
}/*namespace M4D*/

//Include implementation
#include "Imaging/ImageConnection.tcc"

#endif /*_IMAGE_CONNECTION_H*/
