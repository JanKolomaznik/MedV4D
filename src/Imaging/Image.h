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

	int32	minimum;
	int32	maximum;
	float32	elementExtent;

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

	virtual WriterBBoxInterface &
	SetWholeDirtyBBox() = 0;

	virtual const ModificationManager &
	GetModificationManager()const = 0;
protected:
	unsigned		_dimCount;
	DimensionExtents	*_dimensionExtents;
private:

};

class AbstractImage::EWrongDimension
{
	//TODO
};

template< unsigned dim >
class AbstractImageDim;

template<>
class AbstractImageDim< 2 > : public AbstractImage
{
public:
	AbstractImageDim( DimensionExtents *dimExtents ): AbstractImage( 2, dimExtents ) {}
};

template<>
class AbstractImageDim< 3 > : public AbstractImage
{
public:
	AbstractImageDim( DimensionExtents *dimExtents ): AbstractImage( 3, dimExtents ) {}

};

template<>
class AbstractImageDim< 4 > : public AbstractImage
{
public:
	AbstractImageDim( DimensionExtents *dimExtents ): AbstractImage( 4, dimExtents ) {}

};

template< typename ElementType, unsigned dim >
class Image;

/**
 * Partial specialization of image template for two dimensional case.
 **/
template< typename ElementType >
class Image< ElementType, 2 >: public AbstractImageDim< 2 >
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
	 **/
	Image( AbstractImageData::APtr imageData );

	/**
	 * Constructor from typed ImageData - if not possible from some
	 * reason, throwing exception.
	 * \param imageData Dataset storing image data.
	 **/
	Image( typename ImageDataTemplate< ElementType >::Ptr imageData );
	
	~Image();

	/**
	 * \param image Refence to abstract image - predecessor of this class.
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static Image< ElementType, 2 > &
	CastAbstractImage( AbstractImage & image );

	/**
	 * \param image Constant refence to abstract image - predecessor of this class.
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static const Image< ElementType, 2 > &
	CastAbstractImage( const AbstractImage & image );

	/**
	 * \param image Smart pointer to abstract image - predecessor of this class.
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
	GetElement( int32 x, int32 y );

	/**
	 * Access method to data for constant image- checking boundaries.
	 * \param x X coordinate.
	 * \param y Y coordinate.
	 * \exception
	 **/
	const ElementType &
	GetElement( int32 x, int32 y )const;

	ElementType *
	GetPointer( 
			uint32 &width,
			uint32 &height,
			int32 &xStride,
			int32 &yStride
		  )const;

	Ptr
	GetRestricted2DImage( 
			int32 x1, 
			int32 y1, 
			int32 x2, 
			int32 y2 
			);

	WriterBBoxInterface &
	SetDirtyBBox( 
			int32 x1, 
			int32 y1, 
			int32 x2, 
			int32 y2 
			);

	WriterBBoxInterface &
	SetWholeDirtyBBox();

	ReaderBBoxInterface::Ptr
	GetDirtyBBox( 
			int32 x1, 
			int32 y1, 
			int32 x2, 
			int32 y2 
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
class Image< ElementType, 3 >: public AbstractImageDim< 3 >
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
	GetElement( int32 x, int32 y, int32 z );

	const ElementType &
	GetElement( int32 x, int32 y, int32 z )const;

	ElementType *
	GetPointer( 
			uint32 &width,
			uint32 &height,
			uint32 &depth,
			int32 &xStride,
			int32 &yStride,
			int32 &zStride
		  )const;

	typename Image< ElementType, 2 >::Ptr
	GetRestricted2DImage( 
			int32 x1, 
			int32 y1, 
			int32 z1, 
			int32 x2, 
			int32 y2, 
			int32 z2 
			);

	Ptr
	GetRestricted3DImage( 
			int32 x1, 
			int32 y1, 
			int32 z1, 
			int32 x2, 
			int32 y2, 
			int32 z2 
			);

	WriterBBoxInterface &
	SetDirtyBBox( 
			int32 x1, 
			int32 y1, 
			int32 z1, 
			int32 x2, 
			int32 y2, 
			int32 z2 
			);

	WriterBBoxInterface &
	SetWholeDirtyBBox();

	ReaderBBoxInterface::Ptr
	GetDirtyBBox( 
			int32 x1, 
			int32 y1, 
			int32 z1, 
			int32 x2, 
			int32 y2, 
			int32 z2 
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
class Image< ElementType, 4 >: public AbstractImageDim< 4 >
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
	GetElement( int32 x, int32 y, int32 z, int32 t );

	const ElementType &
	GetElement( int32 x, int32 y, int32 z, int32 t )const;

	ElementType *
	GetPointer( 
			uint32 &width,
			uint32 &height,
			uint32 &depth,
			uint32 &time,
			int32 &xStride,
			int32 &yStride,
			int32 &zStride,
			int32 &tStride
		  )const;

	typename Image< ElementType, 2 >::Ptr
	GetRestricted2DImage( 
			int32 x1, 
			int32 y1, 
			int32 z1, 
			int32 t1,
			int32 x2, 
			int32 y2, 
			int32 z2,
			int32 t2
			);

	typename Image< ElementType, 3 >::Ptr
	GetRestricted3DImage( 
			int32 x1, 
			int32 y1, 
			int32 z1, 
			int32 t1,
			int32 x2, 
			int32 y2, 
			int32 z2,
			int32 t2
			);
	Ptr
	GetRestricted4DImage( 
			int32 x1, 
			int32 y1, 
			int32 z1, 
			int32 t1,
			int32 x2, 
			int32 y2, 
			int32 z2,
			int32 t2
			);

	WriterBBoxInterface &
	SetDirtyBBox( 
			int32 x1, 
			int32 y1, 
			int32 z1, 
			int32 t1,
			int32 x2, 
			int32 y2, 
			int32 z2,
			int32 t2
			);

	WriterBBoxInterface &
	SetWholeDirtyBBox();

	ReaderBBoxInterface::Ptr
	GetDirtyBBox( 
			int32 x1, 
			int32 y1, 
			int32 z1, 
			int32 t1,
			int32 x2, 
			int32 y2, 
			int32 z2,
			int32 t2
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
