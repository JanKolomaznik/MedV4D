
#ifndef A_IMAGE_H
#define A_IMAGE_H

#include <memory>
#include "MedV4D/Imaging/ModificationManager.h"
#include "MedV4D/Imaging/ADataset.h"
#include "MedV4D/Common/Vector.h"
#include "MedV4D/Imaging/AImageRegion.h"

namespace M4D
{
/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file AImage.h
 * @{
 **/

namespace Imaging {


/**
 *
 **/
struct DimensionExtents {
	DimensionExtents() :minimum ( 0 ),maximum ( 0 ),elementExtent ( 1.0f ) {}

	int32	minimum;
	int32	maximum;
	float32	elementExtent;

	bool
	operator== ( const DimensionExtents &dimExt ) const {
		return minimum == dimExt.minimum &&
		       maximum == dimExt.maximum &&
		       elementExtent == dimExt.elementExtent;
	}
	bool
	operator!= ( const DimensionExtents &dimExt ) const {
		return ! ( this->operator== ( dimExt ) );
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
class AImage : public ADataset
{
public:
	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO ( AImage )
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO ( ADataset )
	PREPARE_CAST_METHODS_MACRO;
	IS_NOT_CONSTRUCTABLE_MACRO;

	class EBadDimension;

	AImage ( uint16 dim, DimensionExtents *dimExtents );

	virtual
	~AImage() =0;

	const DimensionExtents &
	GetDimensionExtents ( unsigned dimension ) const;


	uint16
	GetDimension() const {
		return _dimCount;
	}

	virtual const int32*
	GetMinimumP() const = 0;

	virtual const int32*
	GetMaximumP() const = 0 ;

	virtual const uint32*
	GetSizeP() const = 0 ;

	virtual const float32*
	GetElementExtentsP() const = 0 ;

	/**
	 * @return ID of element type.
	 **/
	virtual int16
	GetElementTypeID() const=0;

	virtual WriterBBoxInterface &
	SetWholeDirtyBBox() = 0;

	virtual ReaderBBoxInterface::Ptr
	GetWholeDirtyBBox() const = 0;

	virtual const ModificationManager &
	GetModificationManager() const = 0;

	M4D::Common::TimeStamp
	GetEditTimestamp() const {
		return GetModificationManager().GetActualTimestamp();
	}

	virtual AImageRegion::Ptr
	GetAImageRegion() = 0;

	virtual AImageRegion::ConstPtr
	GetAImageRegion() const = 0;
protected:
	uint16			_dimCount;
	DimensionExtents	*_dimensionExtents;
private:

};

class AImage::EBadDimension
{
	//TODO
};

template< size_t Dim >
class AImageDim : public AImage
{
public:
	static const unsigned 		Dimension = Dim;
	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO ( AImageDim< Dimension > )
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO ( AImage )
	PREPARE_CAST_METHODS_MACRO;
	IS_NOT_CONSTRUCTABLE_MACRO;

	typedef Vector< int32, Dimension >	PointType;
	typedef Vector< uint32, Dimension >	SizeType;
	typedef Vector< float32, Dimension >	ElementExtentsType;

	AImageDim ( DimensionExtents *dimExtents ) : AImage ( Dimension, dimExtents ) {
		for ( unsigned i = 0; i < Dimension; ++i ) {
			_minimum[i] = dimExtents[ i ].minimum;
			_maximum[i] = dimExtents[ i ].maximum;
			_size[i] = _maximum[i] - _minimum[i];
		}
	}

	ImageExtentsRecord< Dimension >
	GetImageExtentsRecord()const
	{
		ImageExtentsRecord< Dimension > rec;
		rec.elementExtents = GetElementExtents();
		rec.minimum = GetMinimum();
		rec.maximum = GetMaximum();
		rec.realMinimum = GetRealMinimum();
		rec.realMaximum = GetRealMaximum();
		return rec;
	}

	PointType
	GetMinimum() const {
		return _minimum;
	}

	PointType
	GetMaximum() const {
		return _maximum;
	}

	Vector< float, Dimension >
	GetRealMinimum() const {
		return VectorMemberProduct ( _minimum, _elementExtents );
	}

	Vector< float, Dimension >
	GetRealMaximum() const {
		return VectorMemberProduct ( _maximum, _elementExtents );
	}

	SizeType
	GetSize() const {
		return _size;
	}

	ElementExtentsType
	GetElementExtents() const {
		return _elementExtents;
	}

	const int32*
	GetMinimumP() const {
		return _minimum.data();
	}

	const int32*
	GetMaximumP() const {
		return _maximum.data();
	}

	const uint32*
	GetSizeP() const {
		return _size.data();
	}

	const float32*
	GetElementExtentsP() const {
		return _elementExtents.data();
	}

	AImageRegion::Ptr
	GetAImageRegion() {
		return GetAImageRegionDim();
	}

	virtual typename AImageRegionDim< Dimension >::Ptr
	GetAImageRegionDim() = 0;

	AImageRegion::ConstPtr
	GetAImageRegion() const {
		return GetAImageRegionDim();
	}

	virtual typename AImageRegionDim< Dimension >::ConstPtr
	GetAImageRegionDim() const = 0;

	virtual void
	getChangedRegionSinceTimestamp( PointType &aMinimum, PointType &aMaximum, const Common::TimeStamp &aTimestamp ) const = 0;

protected:
	PointType		_minimum;
	PointType		_maximum;
	SizeType		_size;
	ElementExtentsType	_elementExtents;
};





}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

//include implementation
#include "MedV4D/Imaging/AImage.tcc"

#endif /*A_IMAGE_H*/

