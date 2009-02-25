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
/*template< typename ElementType, unsigned dim >
class Image;*/

/**
 * Partial specialization of image template for two dimensional case.
 **/
template< typename ElementType, unsigned Dim >
class Image: public AbstractImageDim< Dim >
{
public:
	friend class ImageFactory;

	static const unsigned				Dimension = Dim;

	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( Image< ElementType, Dimension > );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( AbstractImageDim< Dimension > );
	PREPARE_CAST_METHODS_MACRO;
	IS_CONSTRUCTABLE_MACRO;

	/**
	 * Dataset contained in this class.
	 **/
	typedef ImageDataTemplate< ElementType >	DataSet;

	/**
	 * Type of elements stored in this image.
	 **/
	typedef ElementType				Element;

	typedef ImageIterator< Element, Dimension >	Iterator;

	typedef ImageRegion< Element, Dimension >	SubRegion;

	typedef Vector< int32, Dimension >		PointType;

	typedef Vector< uint32, Dimension >	SizeType;

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
	static Image< ElementType, Dimension > &
	CastAbstractImage( AbstractImage & image )
		{
			//TODO - handle exception well
			return dynamic_cast< Image< ElementType, Dimension > & >( image );
		}

	/**
	 * \param image Constant refence to abstract image - predecessor of this class.
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static const Image< ElementType, Dimension > &
	CastAbstractImage( const AbstractImage & image )
		{
			//TODO - handle exception well
			return dynamic_cast< const Image< ElementType, Dimension > & >( image );
		}

	/**
	 * \param image Smart pointer to abstract image - predecessor of this class.
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static typename Image< ElementType, Dimension >::Ptr 
	CastAbstractImage( AbstractImage::Ptr & image )
		{
			if( dynamic_cast< Image< ElementType, Dimension > * >( image.get() ) == NULL ) {
				//TODO _THROW_ exception
			}

			return boost::static_pointer_cast< Image< ElementType, Dimension > >( image );
		}
	

	/**
	 * Method used for easy runtime type identification of 
	 * elements types - working only on predefined types.
	 * @return ID of numeric type defined in Common.h
	 **/
	int16
	GetElementTypeID()const
		{ return GetNumericTypeID<ElementType>(); }

	
	inline ElementType &
	GetElement( const PointType &pos );

	inline const ElementType &
	GetElement( const PointType &pos )const;

	/**
	 * Access method to data - checking boundaries.
	 * @param x X coordinate.
	 * @param y Y coordinate.
	 **/
	/*inline ElementType &
	GetElement( int32 x, int32 y );*/

	/**
	 * Access method to data for constant image- checking boundaries.
	 * @param x X coordinate.
	 * @param y Y coordinate.
	 **/
	/*inline const ElementType &
	GetElement( int32 x, int32 y )const;*/

	/*inline ElementType *
	GetPointer( 
			uint32 &width,
			uint32 &height,
			int32 &xStride,
			int32 &yStride
		  )const;*/

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

	/*WriterBBoxInterface &
	SetDirtyBBox( 
			int32 x1, 
			int32 y1, 
			int32 x2, 
			int32 y2 
			);*/

	WriterBBoxInterface &
	SetDirtyBBox( 
			PointType min,
			PointType max 
			);

	WriterBBoxInterface &
	SetWholeDirtyBBox();

	/*ReaderBBoxInterface::Ptr
	GetDirtyBBox( 
			int32 x1, 
			int32 y1, 
			int32 x2, 
			int32 y2 
			)const;*/

	ReaderBBoxInterface::Ptr
	GetDirtyBBox( 
			PointType min,
			PointType max
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
	
	void SerializeClassInfo(OutStream &stream);
	void SerializeProperties(OutStream &stream);
	void SerializeData(OutStream &stream);
	void DeSerializeData(InStream &stream);	
	void DeSerializeProperties(InStream &stream);

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

