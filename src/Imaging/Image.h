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
	class EWrongDimension;
	
	AbstractImage( unsigned dim, DimensionExtents *dimExtents );
	
	virtual
	~AbstractImage()=0;

	const DimensionExtents &
	GetDimensionExtents( unsigned dimension )const;

	unsigned 
	GetDimension()const
		{ return _dimCount; }

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

class AbstractImage::EWrongDimension
{
	//TODO
};


class AbstractImage2D : public AbstractImage
{
public:
	AbstractImage2D( DimensionExtents *dimExtents ): AbstractImage( 2, dimExtents ) {}
};

class AbstractImage3D : public AbstractImage
{
public:
	AbstractImage3D( DimensionExtents *dimExtents ): AbstractImage( 3, dimExtents ) {}

};

class AbstractImage4D : public AbstractImage
{
public:
	AbstractImage4D( DimensionExtents *dimExtents ): AbstractImage( 4, dimExtents ) {}

};

template< typename ElementType, unsigned dim >
class Image;

/**
 * Partial specialization of image template for two dimensional case.
 **/
template< typename ElementType >
class Image< ElementType, 2 >: public AbstractImage2D
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
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static Image< ElementType, 2 > &
	CastAbstractImage( AbstractImage & image );

	/**
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static const Image< ElementType, 2 > &
	CastAbstractImage( const AbstractImage & image );

	/**
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static typename Image< ElementType, 2 >::Ptr 
	CastAbstractImage( AbstractImage::AImagePtr & image );

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
			int &xStride,
			int &yStride
		  )const;

	Ptr
	GetRestricted2DImage( 
			size_t x1, 
			size_t y1, 
			size_t x2, 
			size_t y2 
			);

	WriterBBoxInterface &
	SetDirtyBBox( 
			size_t x1, 
			size_t y1, 
			size_t x2, 
			size_t y2 
			);

	ReaderBBoxInterface::Ptr
	GetDirtyBBox( 
			size_t x1, 
			size_t y1, 
			size_t x2, 
			size_t y2 
			)const;

	const ModificationManager &
	GetModificationManager()const;

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
class Image< ElementType, 3 >: public AbstractImage3D
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
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static Image< ElementType, 3 > &
	CastAbstractImage( AbstractImage & image );

	/**
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static const Image< ElementType, 3 > &
	CastAbstractImage( const AbstractImage & image );

	/**
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static typename Image< ElementType, 3 >::Ptr 
	CastAbstractImage( AbstractImage::AImagePtr & image );

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
			int &xStride,
			int &yStride,
			int &zStride
		  )const;

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

	WriterBBoxInterface &
	SetDirtyBBox( 
			size_t x1, 
			size_t y1, 
			size_t z1, 
			size_t x2, 
			size_t y2, 
			size_t z2 
			);

	ReaderBBoxInterface::Ptr
	GetDirtyBBox( 
			size_t x1, 
			size_t y1, 
			size_t z1, 
			size_t x2, 
			size_t y2, 
			size_t z2 
			)const;


	const ModificationManager &
	GetModificationManager()const;


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
class Image< ElementType, 4 >: public AbstractImage4D
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
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static Image< ElementType, 4 > &
	CastAbstractImage( AbstractImage & image );

	/**
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static const Image< ElementType, 4 > &
	CastAbstractImage( const AbstractImage & image );

	/**
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static typename Image< ElementType, 4 >::Ptr 
	CastAbstractImage( AbstractImage::AImagePtr & image );

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
			int &xStride,
			int &yStride,
			int &zStride,
			int &tStride
		  )const;

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

	WriterBBoxInterface &
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

	ReaderBBoxInterface::Ptr
	GetDirtyBBox( 
			size_t x1, 
			size_t y1, 
			size_t z1, 
			size_t t1,
			size_t x2, 
			size_t y2, 
			size_t z2,
			size_t t2
			)const;

	const ModificationManager &
	GetModificationManager()const;

protected:
	typename ImageDataTemplate< ElementType >::Ptr	_imageData;

	DimensionExtents	_dimExtents[Dimension];
private:
	void
	FillDimensionInfo();

	void
	ReallocateData( typename ImageDataTemplate< ElementType >::Ptr imageData );
};


//Typedefs :

typedef Image< int8, 2 > 	Image2DSigned8b;
typedef Image< uint8, 2 > 	Image2DUnsigned8b;
typedef Image< int16, 2 > 	Image2DSigned16b;
typedef Image< uint16, 2 > 	Image2DUnsigned16b;
typedef Image< int32, 2 > 	Image2DSigned32b;
typedef Image< uint32, 2 > 	Image2DUnsigned32b;

typedef Image< int8, 3 > 	Image3DSigned8b;
typedef Image< uint8, 3 > 	Image3DUnsigned8b;
typedef Image< int16, 3 > 	Image3DSigned16b;
typedef Image< uint16, 3 > 	Image3DUnsigned16b;
typedef Image< int32, 3 > 	Image3DSigned32b;
typedef Image< uint32, 3 > 	Image3DUnsigned32b;


}/*namespace Imaging*/
}/*namespace M4D*/

//include implementation
#include "Imaging/Image.tcc"

#endif /*_IMAGE__H*/
