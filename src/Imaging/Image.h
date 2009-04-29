#ifndef _IMAGE__H
#define _IMAGE__H

#include "Imaging/AImage.h"
#include "Imaging/ImageDataTemplate.h"
#include <boost/shared_ptr.hpp>
#include "Imaging/ModificationManager.h"
#include "Imaging/AbstractDataSet.h"
#include "Imaging/ImageIterator.h"
#include "Imaging/ImageRegion.h"
#include <iostream>

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

	typedef ImageRegion< ElementType, Dimension-1 >	SliceRegion;

	typedef Vector< int32, Dimension >		PointType;

	typedef Vector< uint32, Dimension >	SizeType;

	typedef Vector< float32, Dimension >	ElementExtentsType;

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

	Image( typename ImageDataTemplate< ElementType >::Ptr imageData, PointType minimum, PointType maximum );

	~Image();

	/**
	 * \param image Refence to abstract image - predecessor of this class.
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static Image< ElementType, Dimension > &
	CastAbstractImage( AbstractImage & image )
		{
			try {
				return dynamic_cast< Image< ElementType, Dimension > & >( image );
			} catch (...) {
				_THROW_ ErrorHandling::ExceptionCastProblem();
			}
		}

	/**
	 * \param image Constant refence to abstract image - predecessor of this class.
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static const Image< ElementType, Dimension > &
	CastAbstractImage( const AbstractImage & image )
		{
			try {
				return dynamic_cast< const Image< ElementType, Dimension > & >( image );
			} catch (...) {
				_THROW_ ErrorHandling::ExceptionCastProblem();
			}
		}

	/**
	 * \param image Smart pointer to abstract image - predecessor of this class.
	 * \exception ExceptionCastProblem When casting impossible.
	 **/	
	static typename Image< ElementType, Dimension >::Ptr 
	CastAbstractImage( AbstractImage::Ptr & image )
		{
			if( dynamic_cast< Image< ElementType, Dimension > * >( image.get() ) == NULL ) {
				_THROW_ ErrorHandling::ExceptionCastProblem();
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

	inline ElementType
	GetElementWorldCoords( const Vector< float32, Dim > &pos )const;


	ElementType *
	GetPointer( 
			SizeType &size,
			PointType &strides
		  )const;

	template< unsigned NewDim >
	typename Image< ElementType, NewDim >::Ptr
	GetRestrictedImage( 
			ImageRegion< ElementType, NewDim > region
			)const;

	WriterBBoxInterface &
	SetDirtyBBox( 
			PointType min,
			PointType max 
			);

	WriterBBoxInterface &
	SetWholeDirtyBBox();


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

	SliceRegion
	GetSlice( int32 slice )const;
	
	void SerializeClassInfo(M4D::IO::OutStream &stream);
	void SerializeProperties(M4D::IO::OutStream &stream);
	void SerializeData(M4D::IO::OutStream &stream);
	void DeSerializeData(M4D::IO::InStream &stream);	
	//void DeSerializeProperties(M4D::IO::InStream &stream);

  void Dump(std::ostream &s) const;

	PointType
	GetStrides()const
		{ return _strides;}

	ElementExtentsType 
	GetElementExtents()const
		{ return _elementExtents; }

protected:
	template< unsigned SDim >
	Vector< int32, SDim >
	PosInSource( Vector< int32, Dimension > pos )const
	{
		Vector< int32, SDim > result( _pointerCoordinatesInSource );
		pos -= this->_minimum;
		for( unsigned i=0; i<Dimension; ++i ) {
			result[_dimOrder[i]] += pos[i];
		}
		return result;
	}

	typename ImageDataTemplate< ElementType >::Ptr	_imageData;

	DimensionExtents	_dimExtents[Dimension];
	ElementType		*_pointer;
	PointType		_strides;
	ElementExtentsType	_elementExtents;
	
	///which source dimension is mapped to each dimension of this image
	SizeType			_dimOrder;
	///dimension of source data buffer
	uint32			_sourceDimension;
	///coordinates of point specified by _pointer in source data buffer
	int32			*_pointerCoordinatesInSource;
private:
	void
	FillDimensionInfo();

	void
	ReallocateData( typename ImageDataTemplate< ElementType >::Ptr imageData );

	void
	ReallocateData( typename ImageDataTemplate< ElementType >::Ptr imageData, PointType minimum, PointType maximum );
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

