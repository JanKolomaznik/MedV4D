/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImagePorts.h 
 * @{ 
 **/

#ifndef _IMAGE_PORTS_H
#define _IMAGE_PORTS_H

#include <boost/shared_ptr.hpp>
#include "Imaging/Ports.h"
#include "Imaging/PipelineMessages.h"
#include "Imaging/AbstractDataSet.h"
#include "Imaging/Image.h"
#include <vector>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

//Forward declarations *****************
class AbstractImageConnectionInterface;

template< typename ImageTemplate >
class ImageConnection;

//**************************************

//******************************************************************************
class InputPortAbstractImage: public InputPort
{
public:
	typedef AbstractImageConnectionInterface ConnectionType;

	const AbstractImage&
	GetAbstractImage()const;

	void
	Plug( ConnectionInterface & connection );


	virtual unsigned 
	ImageGetDimension()const
		{ return 0; }

	virtual int
	ImageGetElementTypeID()const
		{ return NTID_UNKNOWN; }

};


class OutputPortAbstractImage: public OutputPort
{
public:
	typedef AbstractImageConnectionInterface ConnectionType;

	AbstractImage&
	GetAbstractImage()const;

	void
	Plug( ConnectionInterface & connection );

	virtual unsigned 
	ImageGetDimension()const
		{ return 0; }

	virtual int
	ImageGetElementTypeID()const
		{ return NTID_UNKNOWN; }
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

	virtual unsigned 
	ImageGetDimension()const
		{ return dimension; }

	virtual int
	ImageGetElementTypeID()const
		{ return GetNumericTypeID< ElementType >(); }

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
			int32 		minimums[ dimension ], 
			int32 		maximums[ dimension ], 
			float32		elementExtents[ dimension ]
		    );

	void
	Plug( ConnectionInterface & connection );

	virtual unsigned 
	ImageGetDimension()const
		{ return dimension; }

	virtual int
	ImageGetElementTypeID()const
		{ return GetNumericTypeID< ElementType >(); }

protected:

};

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#include "Imaging/ImagePorts.tcc"

#endif /*_IMAGE_PORTS_H*/


/** @} */

