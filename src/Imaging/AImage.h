
#ifndef A_IMAGE_H
#define A_IMAGE_H

#include <boost/shared_ptr.hpp>
#include "Imaging/ModificationManager.h"
#include "Imaging/AbstractDataSet.h"
#include "Vector.h"

namespace M4D
{
/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AImage.h 
 * @{ 
 **/

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

	bool
	operator==( const DimensionExtents &dimExt )const
		{ return minimum == dimExt.minimum &&
			maximum == dimExt.maximum &&
			elementExtent == dimExt.elementExtent;
		}
	bool
	operator!=( const DimensionExtents &dimExt )const
		{
			return !(this->operator==( dimExt ));
		}
};

#define IMAGE_TYPE_TEMPLATE_SWITCH_MACRO( AIMAGE_REF, ... ) \
		TYPE_TEMPLATE_SWITCH_MACRO( AIMAGE_REF.GetElementTypeID(), \
				DIMENSION_TEMPLATE_SWITCH_MACRO( AIMAGE_REF.GetDimension(), __VA_ARGS__ ) )



/**
 * Abstract ancestor of image classes. Has access methods 
 * to information about image - dimension, type of elements and proportions.
 * These informations can be used for casting to right type of image or for
 * generic programming. 
 **/
class AbstractImage : public AbstractDataSet
{
public:
	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( AbstractImage );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( AbstractDataSet );
	PREPARE_CAST_METHODS_MACRO;
	IS_NOT_CONSTRUCTABLE_MACRO;
	
		class EBadDimension;
	
	AbstractImage( uint16 dim, DimensionExtents *dimExtents );
	
	virtual
	~AbstractImage()=0;

	const DimensionExtents &
	GetDimensionExtents( unsigned dimension )const;


	uint16 
	GetDimension()const
		{ return _dimCount; }

	/**
	 * @return ID of element type.
	 **/
	virtual int16
	GetElementTypeID()const=0;

	virtual WriterBBoxInterface &
	SetWholeDirtyBBox() = 0;

	virtual ReaderBBoxInterface::Ptr 
	GetWholeDirtyBBox()const = 0;

	virtual const ModificationManager &
	GetModificationManager()const = 0;

	M4D::Common::TimeStamp
	GetEditTimestamp()const
		{ return GetModificationManager().GetActualTimestamp(); }
protected:
	uint16			_dimCount;
	DimensionExtents	*_dimensionExtents;
private:

};

class AbstractImage::EBadDimension
{
	//TODO
};
/**
 * Templated class with specializations for each dimension - now has no special purpose, but in future some 
 * methods from Image classes will be moved here.
 **/
template< unsigned dim >
class AbstractImageDim;

template<>
class AbstractImageDim< 2 > : public AbstractImage
{
public:
	static const unsigned 		Dimension = 2;
	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( AbstractImageDim< Dimension > );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( AbstractImage );
	PREPARE_CAST_METHODS_MACRO;
	IS_NOT_CONSTRUCTABLE_MACRO;


	AbstractImageDim( DimensionExtents *dimExtents ): AbstractImage( 2, dimExtents ) {}

	typedef Vector< int32, Dimension >		PointType;

	PointType
	GetMinimum()const
	{
		PointType result;
		for( unsigned i=0; i < Dimension; ++i ) {
			result[i] = this->GetDimensionExtents(i).minimum;
		}
		return result;
	}

	PointType
	GetMaximum()const
	{
		PointType result;
		for( unsigned i=0; i < Dimension; ++i ) {
			result[i] = this->GetDimensionExtents(i).maximum;
		}
		return result;
	}
};

template<>
class AbstractImageDim< 3 > : public AbstractImage
{
public:
	static const unsigned 		Dimension = 3;
	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( AbstractImageDim< Dimension > );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( AbstractImage );
	PREPARE_CAST_METHODS_MACRO;
	IS_NOT_CONSTRUCTABLE_MACRO;

	AbstractImageDim( DimensionExtents *dimExtents ): AbstractImage( 3, dimExtents ) {}

	typedef Vector< int32, Dimension >		PointType;

	PointType
	GetMinimum()const
	{
		PointType result;
		for( unsigned i=0; i < Dimension; ++i ) {
			result[i] = this->GetDimensionExtents(i).minimum;
		}
		return result;
	}

	PointType
	GetMaximum()const
	{
		PointType result;
		for( unsigned i=0; i < Dimension; ++i ) {
			result[i] = this->GetDimensionExtents(i).maximum;
		}
		return result;
	}
};

template<>
class AbstractImageDim< 4 > : public AbstractImage
{
public:
	static const unsigned 		Dimension = 4;
	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( AbstractImageDim< Dimension > );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( AbstractImage );
	PREPARE_CAST_METHODS_MACRO;
	IS_NOT_CONSTRUCTABLE_MACRO;

	AbstractImageDim( DimensionExtents *dimExtents ): AbstractImage( 4, dimExtents ) {}

	typedef Vector< int32, Dimension >		PointType;

	PointType
	GetMinimum()const
	{
		PointType result;
		for( unsigned i=0; i < Dimension; ++i ) {
			result[i] = this->GetDimensionExtents(i).minimum;
		}
		return result;
	}

	PointType
	GetMaximum()const
	{
		PointType result;
		for( unsigned i=0; i < Dimension; ++i ) {
			result[i] = this->GetDimensionExtents(i).maximum;
		}
		return result;
	}
};



}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

//include implementation
#include "Imaging/AImage.tcc"

#endif /*A_IMAGE_H*/

