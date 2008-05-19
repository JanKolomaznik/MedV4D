#ifndef _IMAGE__H
#define _IMAGE__H

#include "ImageDataTemplate.h"


namespace M4D
{
namespace Imaging
{

struct DimensionExtends
{
	DimensionExtends(){}

	size_t	
};

class AbstractImage
{
public:
	typedef boost::shared_ptr< AbstractImage > APtr;

	AbstractImage(){}
	
	virtual
	~AbstractImage()=0;
protected:

private:

};

template< ElementType, unsigned dim >
class ImageDimensionalTemplate: public AbstractImage 
{
public:
	typedef ElementType		Element;

	ImageDimensionalTemplate( typename ImageDataTemplate< ElementType >::Ptr imageData );
	ImageDimensionalTemplate( AbstractImageData::APtr imageData );
	~ImageDimensionalTemplate();

	const DimensionExtends &
	GetExtends( unsigned dimension );
protected:
	
	ImageDataTemplate< ElementType >	_imageData;
	DimensionExtends					_extends[ dim ];	
private:

};

template< ElementType >
class Image2D: public ImageDimensionalTemplate< ElementType, 2 >
{
public:
	typedef Image2D< ElementType >			ThisClass;
	typedef boost::shared_ptr< ThisClass >	Ptr;

	Image2D( typename ImageDataTemplate< ElementType >::Ptr imageData );
	Image2D( AbstractImageData::APtr imageData );
	~Image2D();

protected:

private:

};


template< ElementType >
class Image3D: public ImageDimensionalTemplate< ElementType, 3 >
{
public:
	typedef Image3D< ElementType >			ThisClass;
	typedef boost::shared_ptr< ThisClass >	Ptr;

	Image3D( typename ImageDataTemplate< ElementType >::Ptr imageData );
	Image3D( AbstractImageData::APtr imageData );
	~Image3D();

protected:

private:

};

template< ElementType >
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


}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*_IMAGE__H*/
