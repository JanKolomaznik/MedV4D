#ifndef _IMAGE__H
#define _IMAGE__H

#include "Imaging/AImage.h"
#include "Imaging/ImageDataTemplate.h"
#include <boost/shared_ptr.hpp>
#include "Imaging/ModificationManager.h"
#include "Imaging/ADataset.h"
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

#define IMAGE_TYPE_PTR_TEMPLATE_CASE_MACRO( AIMAGE_PTR, ... )\
	{ \
		typedef M4D::Imaging::Image< TTYPE, DIM > IMAGE_TYPE; \
		IMAGE_TYPE::Ptr IMAGE = IMAGE_TYPE::Cast( AIMAGE_PTR ); \
		__VA_ARGS__; \
	};

	//usage function< IMAGE_TYPE >( IMAGE )
#define IMAGE_TYPE_PTR_SWITCH_MACRO( AIMAGE_PTR, ... ) \
		TYPE_TEMPLATE_SWITCH_MACRO( (AIMAGE_PTR)->GetElementTypeID(), \
			DIMENSION_TEMPLATE_SWITCH_MACRO( (AIMAGE_PTR)->GetDimension(), IMAGE_TYPE_PTR_TEMPLATE_CASE_MACRO( AIMAGE_PTR, __VA_ARGS__ ) ) )

	//usage function< IMAGE_TYPE >( IMAGE )
#define IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( AIMAGE_PTR, ... ) \
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( (AIMAGE_PTR)->GetElementTypeID(), \
			DIMENSION_TEMPLATE_SWITCH_MACRO( (AIMAGE_PTR)->GetDimension(), IMAGE_TYPE_PTR_TEMPLATE_CASE_MACRO( AIMAGE_PTR, __VA_ARGS__ ) ) )
//*********************************************************************************************
#define IMAGE_TYPE_REF_TEMPLATE_CASE_MACRO( AIMAGE_REF, ... )\
	{ \
		typedef M4D::Imaging::Image< TTYPE, DIM > IMAGE_TYPE; \
		IMAGE_TYPE &IMAGE = IMAGE_TYPE::Cast( AIMAGE_REF ); \
		__VA_ARGS__; \
	};

	//usage function< IMAGE_TYPE >( IMAGE )
#define IMAGE_TYPE_REF_SWITCH_MACRO( AIMAGE_REF, ... ) \
		TYPE_TEMPLATE_SWITCH_MACRO( (AIMAGE_REF).GetElementTypeID(), \
			DIMENSION_TEMPLATE_SWITCH_MACRO( (AIMAGE_REF).GetDimension(), IMAGE_TYPE_REF_TEMPLATE_CASE_MACRO( AIMAGE_REF, __VA_ARGS__ ) ) )

	//usage function< IMAGE_TYPE >( IMAGE )
#define IMAGE_NUMERIC_TYPE_REF_SWITCH_MACRO( AIMAGE_REF, ... ) \
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( (AIMAGE_REF).GetElementTypeID(), \
			DIMENSION_TEMPLATE_SWITCH_MACRO( (AIMAGE_REF).GetDimension(), IMAGE_TYPE_REF_TEMPLATE_CASE_MACRO( AIMAGE_REF, __VA_ARGS__ ) ) )
//*********************************************************************************************
#define IMAGE_TYPE_CONST_REF_TEMPLATE_CASE_MACRO( AIMAGE_REF, ... )\
	{ \
		typedef M4D::Imaging::Image< TTYPE, DIM > IMAGE_TYPE; \
		const IMAGE_TYPE &IMAGE = IMAGE_TYPE::Cast( AIMAGE_REF ); \
		__VA_ARGS__; \
	};

	//usage function< IMAGE_TYPE >( IMAGE )
#define IMAGE_TYPE_CONST_REF_SWITCH_MACRO( AIMAGE_REF, ... ) \
		TYPE_TEMPLATE_SWITCH_MACRO( (AIMAGE_REF).GetElementTypeID(), \
			DIMENSION_TEMPLATE_SWITCH_MACRO( (AIMAGE_REF).GetDimension(), IMAGE_TYPE_CONST_REF_TEMPLATE_CASE_MACRO( AIMAGE_REF, __VA_ARGS__ ) ) )

	//usage function< IMAGE_TYPE >( IMAGE )
#define IMAGE_NUMERIC_TYPE_CONST_REF_SWITCH_MACRO( AIMAGE_REF, ... ) \
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( (AIMAGE_REF).GetElementTypeID(), \
			DIMENSION_TEMPLATE_SWITCH_MACRO( (AIMAGE_REF).GetDimension(), IMAGE_TYPE_CONST_REF_TEMPLATE_CASE_MACRO( AIMAGE_REF, __VA_ARGS__ ) ) )
//*********************************************************************************************


/**
 * Templated class designed for storing raster image data of arbitrary type. 
 * It has specialization for each used dimension.
 * It contains buffer with data, which can be shared among different images - for example 
 * 2D image can share one slice from 3D image. 
 * Sharing is possible because locking is done on buffer and this class has only wrapper methods for locking.
 **/
/*template< typename ElementType, unsigned dim >
class Image;*/

/**
 * Partial specialization of image template for two dimensional case.
 **/
template< typename ElementType, unsigned Dim >
class Image: public AImageDim< Dim >
{
public:
	friend class ImageFactory;

	static const unsigned				Dimension = Dim;

	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( Image< ElementType, Dimension > )
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( AImageDim< Dimension > )
	PREPARE_CAST_METHODS_MACRO;
	IS_CONSTRUCTABLE_MACRO;

	/**
	 * Dataset contained in this class.
	 **/
	typedef ImageDataTemplate< ElementType >	Dataset;

	/**
	 * Type of elements stored in this image.
	 **/
	typedef ElementType				Element;

	/**
	 * Type of iterator, which can iterate over whole image or its parts.
	 **/
	typedef ImageIterator< Element, Dimension >	Iterator;

	/**
	 * Type of image region representation.
	 **/
	typedef ImageRegion< Element, Dimension >	SubRegion;

	/**
	 * Type of image region, which contains only one slice of the original image in arbitrary dimension.
	 **/
	typedef ImageRegion< ElementType, Dimension-1 >	SliceRegion;

	typedef Vector< int32, Dimension >		PointType;

	typedef Vector< uint32, Dimension >		SizeType;

	typedef Vector< float32, Dimension >		ElementExtentsType;

	Image();

	/**
	 * Constructor from AImageData - if not possible from some
	 * reason, throwing exception.
	 * \param imageData Dataset storing image data.
	 **/
	Image( AImageData::APtr imageData );

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
	Cast( AImage & image )
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
	Cast( const AImage & image )
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
	Cast( AImage::Ptr & image )
		{
			if( dynamic_cast< Image< ElementType, Dimension > * >( image.get() ) == NULL ) {
				_THROW_ ErrorHandling::ExceptionCastProblem();
			}

			return boost::static_pointer_cast< Image< ElementType, Dimension > >( image );
		}

	/**
	 * \param image Smart pointer to const abstract image - predecessor of this class.
	 * \exception ExceptionCastProblem When casting impossible.
	 **/
	static typename Image< ElementType, Dimension >::ConstPtr 
	Cast( AImage::ConstPtr & image )
		{
			if( dynamic_cast< const Image< ElementType, Dimension > * >( image.get() ) == NULL ) {
				_THROW_ ErrorHandling::ExceptionCastProblem();
			}

			return boost::static_pointer_cast< const Image< ElementType, Dimension > >( image );
		}
	

	/**
	 * Method used for easy runtime type identification of 
	 * elements types - working only on predefined types.
	 * @return ID of numeric type defined in Common.h
	 **/
	int16
	GetElementTypeID()const
		{ return GetNumericTypeID<ElementType>(); }

	
	/**
	 * Get reference to element on specified position - allow editation of this element.
	 * \param pos Coordinates of desired element
	 * \return Reference to element on given position
	 **/
	inline ElementType &
	GetElement( const PointType &pos );

	/**
	 * Get const reference to element on specified position - only for read.
	 * \param pos Coordinates of desired element
	 * \return Const reference to element on given position
	 **/
	inline const ElementType &
	GetElement( const PointType &pos )const;

	inline ElementType
	GetElementWorldCoords( const Vector< float32, Dim > &pos )const;

	/**
	 * Method for lowlevel access to image data.
	 * \param[out] size Size of the image data in number of elements in each dimension.
	 * \param[out] strides Strides for incrementation of data pointer to move in each dimension
	 * \return Pointer to first element in buffer.
	 **/
	ElementType *
	GetPointer( 
			SizeType &size,
			PointType &strides
		  )const;

	ElementType *
	GetPointer()const;

	inline int
	GetStride( unsigned dim )const
		{
			return _strides[dim];
		}

	inline PointType
	GetStride()const
		{
			return _strides;
		}


	/**
	 * Create image, which is sharing some subregion with this image.
	 * \param[in] region Region, which will be shared among images.
	 * \return Smart pointer to new image.
	 **/
	template< unsigned NewDim >
	typename Image< ElementType, NewDim >::Ptr
	GetRestrictedImage( 
			ImageRegion< ElementType, NewDim > region
			)const;

	/**
	 * \return Should be minimal (or lower) value in the image. 
	 * Without proper initialization or after changing image values it can be outdated or far from actual image minima.
	 **/
	ElementType
	GetLowBand()const;

	/**
	 * \return Should be maximal (or bigger) value in the image. 
	 * Without proper initialization or after changing image values it can be outdated or far from actual image maxima.
	 **/
	ElementType
	GetHighBand()const;

	/**
	 * Mark some part of the image data as dirty - used for synchronization in pipeline filters.
	 * \param[in] min First corner of bounding box of the marked part - included in the bounding box.
	 * \param[in] max Second corner of bounding box of the marked part - NOT included in the bounding box.
	 * \return Handler of dirty state for specified part.
	 **/
	WriterBBoxInterface &
	SetDirtyBBox( 
			PointType min,
			PointType max 
			);

	/**
	 * Mark whole image as dirty - used for synchronization in pipeline filters.
	 * \return Handler of dirty state for specified part - whole image.
	 **/
	WriterBBoxInterface &
	SetWholeDirtyBBox();

	/**
	 * Announce that somebody wants to read some part of image data after all changes on this part were applied - used for synchronization in pipeline filters.
	 * \param[in] min First corner of bounding box of the marked part - included in the bounding box.
	 * \param[in] max Second corner of bounding box of the marked part - NOT included in the bounding box.
	 * \return Handler of read dirty state for specified part.
	 **/
	ReaderBBoxInterface::Ptr
	GetDirtyBBox( 
			PointType min,
			PointType max
			)const;

	/**
	 * Announce that somebody wants to read image data after all changes were applied - used for synchronization in pipeline filters.
	 * \return Handler of read dirty state for image data.
	 **/
	ReaderBBoxInterface::Ptr 
	GetWholeDirtyBBox()const;

	const ModificationManager &
	GetModificationManager()const;

	/**
	 * Get iterator to access all image elements
	 * \return Iterator, which can iterate over whole image
	 **/
	Iterator
	GetIterator()const;

	/**
	 * Get region object for whole image
	 * \return Region representation.
	 **/
	SubRegion
	GetRegion()const;

	typename AImageRegionDim< Dimension >::Ptr
	GetAImageRegionDim()
		{ return typename AImageRegionDim< Dimension >::Ptr( new SubRegion( GetRegion() ) ); }	//TODO -- error handling, effectivenes

	typename AImageRegionDim< Dimension >::ConstPtr
	GetAImageRegionDim()const
		{ return typename AImageRegionDim< Dimension >::ConstPtr( new SubRegion( GetRegion() ) ); }	//TODO -- error handling, effectivenes


	SubRegion
	GetSubRegion(
			PointType min,
			PointType max
			)const;

	SliceRegion
	GetSlice( int32 slice, uint32 perpAxis = Dimension - 1 )const;
	
	void SerializeClassInfo(M4D::IO::OutStream &stream);
	void SerializeProperties(M4D::IO::OutStream &stream);
	void SerializeData(M4D::IO::OutStream &stream) const;
	void DeSerializeData(M4D::IO::InStream &stream);	
	//void DeSerializeProperties(M4D::IO::InStream &stream);

  void Dump(std::ostream &s) const;

	PointType
	GetStrides()const
		{ return _strides;}

	bool
	IsDataContinuous()const
	{
		return _isDataContinuous;
	}

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
	
	ElementType		_minimalValue;
	ElementType		_maximalValue;
	
	///which source dimension is mapped to each dimension of this image
	SizeType			_dimOrder;
	///dimension of source data buffer
	uint32			_sourceDimension;
	///coordinates of point specified by _pointer in source data buffer
	int32			*_pointerCoordinatesInSource;

	bool			_isDataContinuous;
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

