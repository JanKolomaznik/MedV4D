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
struct DimensionExtends
{
	DimensionExtends(){}

//	size_t
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
protected:

private:

};

/**
 *
 **/
template< typename ElementType, unsigned dim >
class Image;/*: public AbstractImage 
{
public:
	typedef Image< ElementType, dim >	ThisClass;
	typedef boost::shared_ptr< ThisClass >	Ptr;
	typedef ElementType			Element;

	Image( 
		typename ImageDataTemplate< ElementType >::Ptr imageData 
		);
	Image( AbstractImageData::APtr imageData );
	~Image();

	const DimensionExtends &
	GetExtends( unsigned dimension );
protected:
	
	ImageDataTemplate< ElementType >	_imageData;
	DimensionExtends			_extends[ dim ];	
private:

};*/

template< typename ElementType >
class Image< ElementType, 2 >: public AbstractImage
{
public:
	typedef Image< ElementType, 2 >		ThisClass;
	typedef boost::shared_ptr< ThisClass >	Ptr;
	typedef ElementType			Element;

	Image( typename ImageDataTemplate< ElementType >::Ptr imageData );
	Image( AbstractImageData::APtr imageData );
	~Image();
	
	ElementType &
	GetElement( size_t x, size_t y );

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

	Image( typename ImageDataTemplate< ElementType >::Ptr imageData );
	Image( AbstractImageData::APtr imageData );
	~Image();
	
	ElementType &
	GetElement( size_t x, size_t y, size_t z );

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

private:


};

/*template< typename ElementType >
class Image2D: public ImageDimensionalTemplate< ElementType, 2 >
{
public:
	typedef Image2D< ElementType >		ThisClass;
	typedef boost::shared_ptr< ThisClass >	Ptr;

	Image2D( typename ImageDataTemplate< ElementType >::Ptr imageData );
	Image2D( AbstractImageData::APtr imageData );
	~Image2D();
	
	ElementType &
	GetElement( size_t x, size_t y );

	Ptr
	GetRestrictedImage2D( 
			size_t x1, 
			size_t y1, 
			size_t x2, 
			size_t y2 
			);

protected:

private:

};


template< typename ElementType >
class Image3D: public ImageDimensionalTemplate< ElementType, 3 >
{
public:
	typedef Image3D< ElementType >		ThisClass;
	typedef boost::shared_ptr< ThisClass >	Ptr;

	Image3D( typename ImageDataTemplate< ElementType >::Ptr imageData );
	Image3D( AbstractImageData::APtr imageData );
	~Image3D();

	ElementType &
	GetElement( size_t x, size_t y, size_t z );

	typename Image2D< ElementType >::Ptr
	GetRestrictedImage2D( 
			size_t x1, 
			size_t y1, 
			size_t z1, 
			size_t x2, 
			size_t y2, 
			size_t z2 
			);

	Ptr
	GetRestrictedImage3D( 
			size_t x1, 
			size_t y1, 
			size_t z1, 
			size_t x2, 
			size_t y2, 
			size_t z2 
			);
protected:

private:

};

template< typename ElementType >
class Image4D: public ImageDimensionalTemplate< ElementType, 4 >
{
public:
	typedef Image4D< ElementType >			ThisClass;
	typedef boost::shared_ptr< ThisClass >	Ptr;

	Image4D( typename ImageDataTemplate< ElementType >::Ptr imageData );
	Image4D( AbstractImageData::APtr imageData );
	~Image4D();

protected:

private:

};
*/

}/*namespace Imaging*/
}/*namespace M4D*/

//include implementation
#include "Imaging/Image.tcc"

#endif /*_IMAGE__H*/
