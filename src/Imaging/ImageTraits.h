/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageTraits.h 
 * @{ 
 **/

#ifndef _IMAGE_TRAITS_H
#define _IMAGE_TRAITS_H

#include "Imaging/AbstractDataSet.h"
#include "Imaging/Image.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

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

	typedef	AbstractImage 		ClassName;
	typedef ConnectionTyped< ClassName > 	Connection;
	typedef ConnectionInterfaceTyped< ClassName > 	IConnection;
	typedef InputPortTyped< ClassName > 	InputPort;
	typedef OutputPortTyped< ClassName > 	OutputPort;
	typedef boost::shared_ptr< ClassName >	Ptr;
};

template< unsigned Dim >
class ImageTraits< AbstractImageDim< Dim > >
{
public:
	typedef void	ElementType;

	static const unsigned	Dimension = Dim;

	typedef	AbstractImageDim< Dim > 		ClassName;
	typedef ConnectionTyped< ClassName > 	Connection;
	typedef ConnectionInterfaceTyped< ClassName > 	IConnection;
	typedef InputPortTyped< ClassName > 	InputPort;
	typedef OutputPortTyped< ClassName > 	OutputPort;
	typedef boost::shared_ptr< ClassName >	Ptr;
};

template< typename EType, unsigned Dim > 
class ImageTraits< Image< EType, Dim > >
{
public:
	typedef EType	ElementType;

	static const unsigned	Dimension = Dim;
	typedef	Image< EType, Dim > 			ClassName;
	typedef ConnectionTyped< ClassName > 	Connection;
	typedef ConnectionInterfaceTyped< ClassName > 	IConnection;
	typedef InputPortTyped< ClassName > 	InputPort;
	typedef OutputPortTyped< ClassName > 	OutputPort;
	typedef boost::shared_ptr< ClassName >	Ptr;
};

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_IMAGE_TRAITS_H*/

/** @} */

