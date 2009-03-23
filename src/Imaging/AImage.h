
#ifndef A_IMAGE_H
#define A_IMAGE_H

#include <boost/shared_ptr.hpp>
#include "Imaging/ModificationManager.h"
#include "Imaging/AbstractDataSet.h"
#include "common/Vector.h"

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

template< unsigned Dim >
class AbstractImageDim : public AbstractImage
{
public:
	static const unsigned 		Dimension = Dim;
	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( AbstractImageDim< Dimension > );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( AbstractImage );
	PREPARE_CAST_METHODS_MACRO;
	IS_NOT_CONSTRUCTABLE_MACRO;

	typedef Vector< int32, Dimension >	PointType;
	typedef Vector< uint32, Dimension >	SizeType;

	AbstractImageDim( DimensionExtents *dimExtents ): AbstractImage( Dimension, dimExtents ) 
		{
			for( unsigned i = 0; i < Dimension; ++i ) {
				_minimum[i] = dimExtents[ i ].minimum;
				_maximum[i] = dimExtents[ i ].maximum;
				_size[i] = _maximum[i] - _minimum[i];
			}
		}


	PointType
	GetMinimum()const
		{ return _minimum;}

	PointType
	GetMaximum()const
		{ return _maximum; }


protected:
	PointType	_minimum;
	PointType	_maximum;
	SizeType	_size;
};





}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

//include implementation
#include "Imaging/AImage.tcc"

#endif /*A_IMAGE_H*/

