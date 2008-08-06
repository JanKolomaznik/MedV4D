#ifndef _IMAGE_PORTS_H
#define _IMAGE_PORTS_H

#include <boost/shared_ptr.hpp>
#include "Imaging/AbstractDataSet.h"
#include "Imaging/Image.h"
#include <vector>
#include "Imaging/PipelineMessages.h"
#include "Imaging/Ports.h"

namespace M4D
{
namespace Imaging
{

//Forward declarations *****************
//class AbstractImageConnection;

//template< typename ImageTemplate >
//class ImageConnection;

//**************************************

//******************************************************************************
class InputPortAbstractImage: public InputPort
{
public:
	typedef AbstractImageConnection ConnectionType;

	const AbstractImage&
	GetAbstractImage()const;

	void
	Plug( ConnectionInterface & connection );

protected:
};


class OutputPortAbstractImage: public OutputPort
{
public:
	typedef AbstractImageConnection ConnectionType;

	AbstractImage&
	GetAbstractImage()const;

	void
	Plug( ConnectionInterface & connection );
};
//******************************************************************************
template< typename ImageType >
class InputPortImageFilter;

template< typename ElementType, unsigned dimension >
class InputPortImageFilter< Image< ElementType, dimension > >: public InputPortAbstractImage
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > ImageType;
	typedef typename M4D::Imaging::ImageConnection< ImageType > ConnectionType;

	InputPortImageFilter() {}

	const ImageType&
	GetImage()const;
	
	void
	Plug( ConnectionInterface & connection );

	/*void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		);*/


protected:
	
};

template< typename ImageType >
class OutputPortImageFilter;

template< typename ElementType, unsigned dimension >
class OutputPortImageFilter< Image< ElementType, dimension > >: public OutputPortAbstractImage
{
public:
	typedef typename M4D::Imaging::Image< ElementType, dimension > ImageType;
	typedef typename M4D::Imaging::ImageConnection< ImageType > ConnectionType;

	OutputPortImageFilter() {}

	//TODO - check const modifier
	ImageType&
	GetImage()const;

	void
	SetImageSize( 
			size_t 		minimums[ dimension ], 
			size_t 		maximums[ dimension ], 
			float32		elementExtents[ dimension ]
		    );

	void
	Plug( ConnectionInterface & connection );

	/*void
	PlugTyped( ImageConnection< ImageType > & connection );*/
	
	/*void
	SendMessage( 
		PipelineMessage::Ptr 			msg, 
		PipelineMessage::MessageSendStyle 	sendStyle 
		);*/

protected:

};

}/*namespace Imaging*/
}/*namespace M4D*/

#include "Imaging/ImagePorts.tcc"

#endif /*_IMAGE_PORTS_H*/

