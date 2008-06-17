#ifndef _IMAGE__H
#define _IMAGE__H

#include "Imaging/ImageDataTemplate.h"
#include <boost/shared_ptr.hpp>
#include "Imaging/ModificationManager.h"
#include "Imaging/AbstractDataSet.h"

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
 * Abstract ancestor of image classes. Have virtual accesing methods 
 * to information about dataset - useful for casting to right type.
 **/
class AbstractImage : public AbstractDataSet
{
public:
	/**
	 * Smart pointer to this class.
	 **/
	typedef boost::shared_ptr< AbstractImage > AImagePtr;

	AbstractImage( unsigned dim, DimensionExtents *dimExtents );
	
	virtual
	~AbstractImage()=0;

	const DimensionExtents &
	GetDimensionExtents( unsigned dimension )const;

	/**
	 * @return ID of element type.
	 **/
	virtual int
	GetElementTypeID()const=0;
protected:
	unsigned		_dimCount;
	DimensionExtents	*_dimensionExtents;
private:

};

template< typename ElementType, unsigned dim >
class Image;

/**
 * Partial specialization of image template for two dimensional case.
 **/
template< typename ElementType >
class Image< ElementType, 2 >: public AbstractImage
{
public:
	friend class ImageFactory;
	/**
	 * Type of "this" class.
	 **/
	typedef Image< ElementType, 2 >			ThisClass;

	/**
	 * Dataset contained in this class.
	 **/
	typedef ImageDataTemplate< ElementType >	DataSet;

	/**
	 * Smart pointer type for this class.
	 **/
	typedef boost::shared_ptr< ThisClass >		Ptr;

	/**
	 * Type of elements stored in this image.
	 **/
	typedef ElementType				Element;

	static const unsigned	Dimension = 2;


	Image();

	/**
	 * Constructor from AbstractImageData - if not possible from some
	 * reason, throwing exception.
	 * \param imageData Dataset storing image data.
	 * \exception
	 **/
	Image( AbstractImageData::APtr imageData );

	/**
	 * Constructor from typed ImageData - if not possible from some
	 * reason, throwing exception.
	 * \param imageData Dataset storing image data.
	 * \exception
	 **/
	Image( typename ImageDataTemplate< ElementType >::Ptr imageData );
	
	~Image();
	
	/**
	 * Method used for easy runtime type identification of 
	 * elements types - working only on predefined types.
	 * @return ID of numeric type defined in Common.h
	 **/
	int
	GetElementTypeID()const
		{ return GetNumericTypeID<ElementType>(); }

	/**
	 * Access method to data - checking boundaries.
	 * \param x X coordinate.
	 * \param y Y coordinate.
	 * \exception
	 **/
	ElementType &
	GetElement( size_t x, size_t y );

	/**
	 * Access method to data for constant image- checking boundaries.
	 * \param x X coordinate.
	 * \param y Y coordinate.
	 * \exception
	 **/
	const ElementType &
	GetElement( size_t x, size_t y )const;

	ElementType *
	GetPointer( 
			size_t &width,
			size_t &height,
			size_t &xStride,
			size_t &yStride
		  );

	Ptr
	GetRestricted2DImage( 
			size_t x1, 
			size_t y1, 
			size_t x2, 
			size_t y2 
			);

	ModificationBBox2D &
	SetDirtyBBox( 
			size_t x1, 
			size_t y1, 
			size_t x2, 
			size_t y2 
			);
protected:
	typename ImageDataTemplate< ElementType >::Ptr	_imageData;

	DimensionExtents	_dimExtents[Dimension];
private:
	void
	FillDimensionInfo();

	void
	ReallocateData( typename ImageDataTemplate< ElementType >::Ptr imageData );

};

/**
 * Partial specialization of image template for three dimensional case.
 **/
template< typename ElementType >
class Image< ElementType, 3 >: public AbstractImage
{
public:
	friend class ImageFactory;
	/**
	 * Type of "this" class.
	 **/
	typedef Image< ElementType, 3 >		ThisClass;

	/**
	 * Dataset contained in this class.
	 **/
	typedef ImageDataTemplate< ElementType >	DataSet;

	/**
	 * Smart pointer type for this class.
	 **/
	typedef boost::shared_ptr< ThisClass >	Ptr;

	/**
	 * Type of elements stored in this image.
	 **/
	typedef ElementType			Element;

	static const unsigned	Dimension = 3;

	Image();

	Image( AbstractImageData::APtr imageData );

	Image( typename ImageDataTemplate< ElementType >::Ptr imageData );

	~Image();

	/**
	 * Method used for easy runtime type identification of 
	 * elements types - working only on predefined types.
	 * @return ID of numeric type defined in Common.h
	 **/
	int
	GetElementTypeID()const
		{ return GetNumericTypeID<ElementType>(); }

	ElementType &
	GetElement( size_t x, size_t y, size_t z );

	const ElementType &
	GetElement( size_t x, size_t y, size_t z )const;

	ElementType *
	GetPointer( 
			size_t &width,
			size_t &height,
			size_t &depth,
			size_t &xStride,
			size_t &yStride,
			size_t &zStride
		  );

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

	ModificationBBox3D &
	SetDirtyBBox( 
			size_t x1, 
			size_t y1, 
			size_t z1, 
			size_t x2, 
			size_t y2, 
			size_t z2 
			);

protected:
	typename ImageDataTemplate< ElementType >::Ptr	_imageData;

	DimensionExtents	_dimExtents[Dimension];
private:
	void
	FillDimensionInfo();

	void
	ReallocateData( typename ImageDataTemplate< ElementType >::Ptr imageData );
};

/**
 * Partial specialization of image template for four dimensional case.
 **/
template< typename ElementType >
class Image< ElementType, 4 >: public AbstractImage
{
public:
	friend class ImageFactory;
	/**
	 * Type of "this" class.
	 **/
	typedef Image< ElementType, 4 >		ThisClass;

	/**
	 * Dataset contained in this class.
	 **/
	typedef ImageDataTemplate< ElementType >	DataSet;

	/**
	 * Smart pointer type for this class.
	 **/
	typedef boost::shared_ptr< ThisClass >	Ptr;

	/**
	 * Type of elements stored in this image.
	 **/
	typedef ElementType			Element;

	static const unsigned	Dimension = 4;

	Image();

	Image( AbstractImageData::APtr imageData );

	Image( typename ImageDataTemplate< ElementType >::Ptr imageData );
	
	~Image();
	
	/**
	 * Method used for easy runtime type identification of 
	 * elements types - working only on predefined types.
	 * @return ID of numeric type defined in Common.h
	 **/
	int
	GetElementTypeID()const
		{ return GetNumericTypeID<ElementType>(); }

	ElementType &
	GetElement( size_t x, size_t y, size_t z, size_t t );

	const ElementType &
	GetElement( size_t x, size_t y, size_t z, size_t t )const;

	ElementType *
	GetPointer( 
			size_t &width,
			size_t &height,
			size_t &depth,
			size_t &time,
			size_t &xStride,
			size_t &yStride,
			size_t &zStride,
			size_t &tStride
		  );

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

	ModificationBBox3D &
	SetDirtyBBox( 
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
	typename ImageDataTemplate< ElementType >::Ptr	_imageData;

	DimensionExtents	_dimExtents[Dimension];
private:
	void
	FillDimensionInfo();

	void
	ReallocateData( typename ImageDataTemplate< ElementType >::Ptr imageData );
};


}/*namespace Imaging*/
}/*namespace M4D*/

//include implementation
#include "Imaging/Image.tcc"

#endif /*_IMAGE__H*/
