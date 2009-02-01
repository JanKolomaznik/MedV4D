#ifndef _IMAGE__H
#define _IMAGE__H

#include "Imaging/AImage.h"
#include "Imaging/ImageDataTemplate.h"
#include <boost/shared_ptr.hpp>
#include "Imaging/ModificationManager.h"
#include "Imaging/AbstractDataSet.h"
#include "Imaging/ImageIterator.h"
#include "Imaging/ImageRegion.h"

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Image.h 
 * @{ 
 **/

namespace Imaging
{

#define IMAGE_TYPE_TEMPLATE_CASE_MACRO( AIMAGE_PTR, ... )\
	{ \
		typedef M4D::Imaging::Image< TTYPE, DIM > IMAGE_TYPE; \
		IMAGE_TYPE::Ptr IMAGE = IMAGE_TYPE::CastAbstractImage( AIMAGE_PTR ); \
		__VA_ARGS__; \
	};

	//usage function< IMAGE_TYPE >( IMAGE )
#define IMAGE_TYPE_PTR_SWITCH_MACRO( AIMAGE_PTR, ... ) \
		TYPE_TEMPLATE_SWITCH_MACRO( AIMAGE_PTR->GetElementTypeID(), \
			DIMENSION_TEMPLATE_SWITCH_MACRO( AIMAGE_PTR->GetDimension(), IMAGE_TYPE_TEMPLATE_CASE_MACRO( AIMAGE_PTR, __VA_ARGS__ ) ) )

	//usage function< IMAGE_TYPE >( IMAGE )
#define IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( AIMAGE_PTR, ... ) \
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( AIMAGE_PTR->GetElementTypeID(), \
			DIMENSION_TEMPLATE_SWITCH_MACRO( AIMAGE_PTR->GetDimension(), IMAGE_TYPE_TEMPLATE_CASE_MACRO( AIMAGE_PTR, __VA_ARGS__ ) ) )
/**
 * Templated class made for storing raster image data of certain type. 
 * It has specialization for each used dimension.
 * It contains buffer with data, which can be shared among different images - for example 
 * 2D image can share one slice from 3D image. But now this sharing concept isn't finished. And will be available in
 * future versions. 
 * Sharing is possible because locking is done on buffer and this class has only wrapper methods for locking.
 **/
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
	typedef AbstractImageDim< 2 >			PredecessorType;

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

	static const unsigned				Dimension = 2;

	typedef ImageIterator< Element, Dimension >	Iterator;

	typedef ImageRegion< Element, Dimension >	SubRegion;

	typedef Coordinates< int32, Dimension >		PointType;

	typedef Coordinates< uint32, Dimension >	SizeType;

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
	
	Image( typename ImageDataTemplate< ElementType >::Ptr imageData, SubRegion region );

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
	int16
	GetElementTypeID()const
		{ return GetNumericTypeID<ElementType>(); }

	/**
	 * Access method to data - checking boundaries.
	 * @param x X coordinate.
	 * @param y Y coordinate.
	 **/
	inline ElementType &
	GetElement( int32 x, int32 y );

	/**
	 * Access method to data for constant image- checking boundaries.
	 * @param x X coordinate.
	 * @param y Y coordinate.
	 **/
	inline const ElementType &
	GetElement( int32 x, int32 y )const;

	inline ElementType *
	GetPointer( 
			uint32 &width,
			uint32 &height,
			int32 &xStride,
			int32 &yStride
		  )const;

	ElementType *
	GetPointer( 
			SizeType &size,
			PointType &strides
		  )const;

	template< unsigned NewDim >
	typename Image< ElementType, NewDim >::Ptr
	GetRestrictedImage( 
			ImageRegion< ElementType, NewDim > region
			);
	/*
	typename Image< ElementType, 2 >::Ptr
	GetRestricted2DImage( 
			ImageRegion< ElementType, 2 > region
			);
	*/
	/*Ptr
	GetRestricted2DImage( 
			int32 x1, 
			int32 y1, 
			int32 x2, 
			int32 y2 
			);*/

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

	ReaderBBoxInterface::Ptr 
	GetWholeDirtyBBox()const;

	const ModificationManager &
	GetModificationManager()const;

	Iterator
	GetIterator()const;

	SubRegion
	GetRegion()const;

	SubRegion
	GetSubRegion(
			PointType min,
			PointType max
			)const;
	
	void Serialize(iAccessStream &stream);
	void DeSerialize(iAccessStream &stream);

  void Dump(void);

protected:
	typename ImageDataTemplate< ElementType >::Ptr	_imageData;

	DimensionExtents	_dimExtents[Dimension];
	ElementType		*_pointer;
	
	uint32			_dimOrder[ Dimension ];
	uint32			_sourceDimension;
	int32			*_pointerCoordinatesInSource;
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
	typedef AbstractImageDim< 3 >		PredecessorType;

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

	typedef ImageIterator< Element, Dimension >	Iterator;

	typedef ImageRegion< Element, Dimension >	SubRegion;

	typedef Coordinates< int, Dimension >		PointType;

	typedef Coordinates< uint32, Dimension >	SizeType;

	Image();

	Image( AbstractImageData::APtr imageData );

	Image( typename ImageDataTemplate< ElementType >::Ptr imageData );

	Image( typename ImageDataTemplate< ElementType >::Ptr imageData, SubRegion region );

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
	int16
	GetElementTypeID()const
		{ return GetNumericTypeID<ElementType>(); }

	inline ElementType &
	GetElement( int32 x, int32 y, int32 z );

	inline const ElementType &
	GetElement( int32 x, int32 y, int32 z )const;

	inline ElementType *
	GetPointer( 
			uint32 &width,
			uint32 &height,
			uint32 &depth,
			int32 &xStride,
			int32 &yStride,
			int32 &zStride
		  )const;

	ElementType *
	GetPointer( 
			SizeType &size,
			PointType &strides
		  )const;

	template< unsigned NewDim >
	typename Image< ElementType, NewDim >::Ptr
	GetRestrictedImage( 
			ImageRegion< ElementType, NewDim > region
			);

	/*
	typename Image< ElementType, 2 >::Ptr
	GetRestricted2DImage( 
			ImageRegion< ElementType, 2 > region
			);

	typename Image< ElementType, 3 >::Ptr
	GetRestricted3DImage( 
			ImageRegion< ElementType, 3 > region
			);
	*/
/*	typename Image< ElementType, 2 >::Ptr
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
*/
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

	ReaderBBoxInterface::Ptr 
	GetWholeDirtyBBox()const;


	const ModificationManager &
	GetModificationManager()const;

	Iterator
	GetIterator()const;

	SubRegion
	GetRegion()const;

	SubRegion
	GetSubRegion( 
			PointType min,
			PointType max
			)const;
	
	void Serialize(iAccessStream &stream);
	void DeSerialize(iAccessStream &stream);

  void Dump(void);

protected:
	typename ImageDataTemplate< ElementType >::Ptr	_imageData;

	DimensionExtents	_dimExtents[Dimension];
	ElementType		*_pointer;
	
	uint32			_dimOrder[ Dimension ];
	uint32			_sourceDimension;
	int32			*_pointerCoordinatesInSource;
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
	typedef AbstractImageDim< 4 >		PredecessorType;

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

	typedef ImageIterator< Element, Dimension >	Iterator;

	typedef ImageRegion< Element, Dimension >	SubRegion;

	typedef Coordinates< int, Dimension >		PointType;

	typedef Coordinates< uint32, Dimension >	SizeType;

	Image();

	Image( AbstractImageData::APtr imageData );

	Image( typename ImageDataTemplate< ElementType >::Ptr imageData );
	
	Image( typename ImageDataTemplate< ElementType >::Ptr imageData, SubRegion region );

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
	int16
	GetElementTypeID()const
		{ return GetNumericTypeID<ElementType>(); }

	inline ElementType &
	GetElement( int32 x, int32 y, int32 z, int32 t );

	inline const ElementType &
	GetElement( int32 x, int32 y, int32 z, int32 t )const;

	inline ElementType *
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

	ElementType *
	GetPointer( 
			SizeType &size,
			PointType &strides
		  )const;

	typename Image< ElementType, 2 >::Ptr
	GetRestricted2DImage( 
			ImageRegion< ElementType, 2 > region
			);
			/*int32 x1, 
			int32 y1, 
			int32 z1, 
			int32 t1,
			int32 x2, 
			int32 y2, 
			int32 z2,
			int32 t2
			);*/

	typename Image< ElementType, 3 >::Ptr
	GetRestricted3DImage( 
			ImageRegion< ElementType, 3 > region
			);
			/*int32 x1, 
			int32 y1, 
			int32 z1, 
			int32 t1,
			int32 x2, 
			int32 y2, 
			int32 z2,
			int32 t2
			);*/
	Ptr
	GetRestricted4DImage( 
			ImageRegion< ElementType, 2 > region
			);
			/*int32 x1, 
			int32 y1, 
			int32 z1, 
			int32 t1,
			int32 x2, 
			int32 y2, 
			int32 z2,
			int32 t2
			);*/

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

	Iterator
	GetIterator()const;

	SubRegion
	GetRegion()const;

	SubRegion
	GetSubRegion(
			PointType min,
			PointType max
			)const;
	
	void Serialize(iAccessStream &stream);
	void DeSerialize(iAccessStream &stream);

  void Dump(void);

protected:
	typename ImageDataTemplate< ElementType >::Ptr	_imageData;

	DimensionExtents	_dimExtents[Dimension];
	ElementType		*_pointer;
	
	uint32			_dimOrder[ Dimension ];
	uint32			_sourceDimension;
	int32			*_pointerCoordinatesInSource;
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

typedef Image< uint8, 2 >	Mask2D;
typedef Image< uint8, 3 >	Mask3D;

}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

//include implementation
#include "Imaging/Image.tcc"

#endif /*_IMAGE__H*/

