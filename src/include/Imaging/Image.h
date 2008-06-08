#ifndef _IMAGE__H
#define _IMAGE__H

#include "Imaging/ImageDataTemplate.h"
#include <boost/shared_ptr.hpp>


namespace M4D
{
namespace Imaging
{

/**
 *
 **/
struct DimensionExtents
{
	DimensionExtents():minimum(0),maximum(0),elementExtent(1.0f){}

	size_t	minimum;
	size_t	maximum;
	float	elementExtent;

};

/**
 *
 **/
class AbstractImage : public AbstractDataSet
{
public:
	typedef boost::shared_ptr< AbstractImage > AImagePtr;

	AbstractImage(){}
	
	virtual
	~AbstractImage()=0;

	const DimensionExtents &
	GetDimensionExtents( unsigned dimension );
protected:

private:

};

/**
 *
 **/
template< typename ElementType, unsigned dim >
class Image;

template< typename ElementType >
class Image< ElementType, 2 >: public AbstractImage
{
public:
	typedef Image< ElementType, 2 >		ThisClass;
	typedef boost::shared_ptr< ThisClass >	Ptr;
	typedef ElementType			Element;

	Image( AbstractImageData::APtr imageData );

	Image( typename ImageDataTemplate< ElementType >::Ptr imageData );
	
	~Image();
	
	ElementType &
	GetElement( size_t x, size_t y );

	const ElementType &
	GetElement( size_t x, size_t y )const;

	Ptr
	GetRestricted2DImage( 
			size_t x1, 
			size_t y1, 
			size_t x2, 
			size_t y2 
			);

protected:

private:


};

template< typename ElementType >
class Image< ElementType, 3 >: public AbstractImage
{
public:
	typedef Image< ElementType, 3 >		ThisClass;
	typedef boost::shared_ptr< ThisClass >	Ptr;
	typedef ElementType			Element;

	Image( AbstractImageData::APtr imageData );

	Image( typename ImageDataTemplate< ElementType >::Ptr imageData );

	~Image();
	
	ElementType &
	GetElement( size_t x, size_t y, size_t z );

	const ElementType &
	GetElement( size_t x, size_t y, size_t z )const;

	typename Image< ElementType, 2 >::Ptr
	GetRestricted2DImage( 
			size_t x1, 
			size_t y1, 
			size_t z1, 
			size_t x2, 
			size_t y2, 
			size_t z2 
			);

	Ptr
	GetRestricted3DImage( 
			size_t x1, 
			size_t y1, 
			size_t z1, 
			size_t x2, 
			size_t y2, 
			size_t z2 
			);
protected:
	ImageDataTemplate< ElementType >	_data;

private:


};

template< typename ElementType >
class Image< ElementType, 4 >: public AbstractImage
{
public:
	typedef Image< ElementType, 4 >		ThisClass;
	typedef boost::shared_ptr< ThisClass >	Ptr;
	typedef ElementType			Element;

	Image( AbstractImageData::APtr imageData );

	Image( typename ImageDataTemplate< ElementType >::Ptr imageData );
	
	~Image();
	
	ElementType &
	GetElement( size_t x, size_t y, size_t z, size_t t );

	const ElementType &
	GetElement( size_t x, size_t y, size_t z, size_t t )const;

	typename Image< ElementType, 2 >::Ptr
	GetRestricted2DImage( 
			size_t x1, 
			size_t y1, 
			size_t z1, 
			size_t t1,
			size_t x2, 
			size_t y2, 
			size_t z2,
			size_t t2
			);

	typename Image< ElementType, 3 >::Ptr
	GetRestricted3DImage( 
			size_t x1, 
			size_t y1, 
			size_t z1, 
			size_t t1,
			size_t x2, 
			size_t y2, 
			size_t z2,
			size_t t2
			);
	Ptr
	GetRestricted4DImage( 
			size_t x1, 
			size_t y1, 
			size_t z1, 
			size_t t1,
			size_t x2, 
			size_t y2, 
			size_t z2,
			size_t t2
			);
protected:

private:


};


}/*namespace Imaging*/
}/*namespace M4D*/

//include implementation
#include "Imaging/Image.tcc"

#endif /*_IMAGE__H*/
