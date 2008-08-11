#ifndef _IMAGE_TRAITS_H
#define _IMAGE_TRAITS_H

#include "Imaging/AbstractDataSet.h"
#include "Imaging/Image.h"

namespace M4D
{
namespace Imaging
{

template< typename ImageType >
class ImageTraits;

template<>
class ImageTraits< AbstractImage >
{
public:
	typedef void	ElementType;

	static const unsigned	Dimension = 0;

	typedef AbstractImageConnection		Connection;
	typedef InputPortAbstractImage 		InputPort;
	typedef OutputPortAbstractImage 		OutputPort;
	typedef boost::shared_ptr< AbstractImage >	Ptr;
};

template< typename EType, unsigned Dim > 
class ImageTraits< Image< EType, Dim > >
{
public:
	typedef EType	ElementType;

	static const unsigned	Dimension = Dim;

	typedef typename M4D::Imaging::ImageConnection< M4D::Imaging::Image< EType, Dim > > 		Connection;
	typedef typename M4D::Imaging::InputPortImageFilter< M4D::Imaging::Image< EType, Dim > > 	InputPort;
	typedef typename M4D::Imaging::OutputPortImageFilter< M4D::Imaging::Image< EType, Dim > > 	OutputPort;
	typedef typename boost::shared_ptr< M4D::Imaging::Image< EType, Dim > >				Ptr;
};

}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*_IMAGE_TRAITS_H*/
