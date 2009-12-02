/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ImageTraits.h 
 * @{ 
 **/

#ifndef _IMAGE_TRAITS_H
#define _IMAGE_TRAITS_H

#include "Imaging/ADataset.h"
#include "Imaging/Image.h"
#include "common/Types.h"
#include "Imaging/ConnectionInterface.h"
#include "Imaging/Ports.h"

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
class ImageTraits< AImage >
{
public:
	typedef void	ElementType;

	static const unsigned	Dimension = 0;

	typedef	AImage 		ClassName;
	typedef ConnectionTyped< ClassName > 	Connection;
	typedef ConnectionInterfaceTyped< ClassName > 	IConnection;
	typedef InputPortTyped< ClassName > 	InputPort;
	typedef OutputPortTyped< ClassName > 	OutputPort;
	typedef boost::shared_ptr< ClassName >	Ptr;
};

template< unsigned Dim >
class ImageTraits< AImageDim< Dim > >
{
public:
	typedef void	ElementType;

	static const unsigned	Dimension = Dim;

	typedef	AImageDim< Dim > 		ClassName;
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

