#ifndef _AIMAGE_REGION_H
#define _AIMAGE_REGION_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/Vector.h"
#include "MedV4D/Imaging/DatasetDefinitionTools.h"

namespace M4D
{
/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file AImageRegion.h
 * @{
 **/

namespace Imaging {


template< unsigned tDim >
struct ImageExtentsRecord
{
	static const unsigned 	Dimension = tDim;
	
	Vector< float32, Dimension > realMinimum;
	Vector< float32, Dimension > realMaximum;
	Vector< int32, Dimension > minimum;
	Vector< int32, Dimension > maximum;
	
	Vector< float32, Dimension > elementExtents;
};
	
class AImageRegion
{


public:
        STANDARD_DECLARATIONS_MACRO ( AImageRegion )

        virtual ~AImageRegion() {}

        virtual AImageRegion::Ptr
        Clone() = 0;

        virtual AImageRegion::ConstPtr
        Clone() const = 0;

        virtual uint32
        GetDimension() const = 0;

        virtual int16
        GetElementTypeID() const = 0;
};

template< unsigned Dim >
class AImageRegionDim: public AImageRegion
{
public:
        STANDARD_DECLARATIONS_MACRO ( AImageRegionDim< Dim > )

        CONFIGURABLE_PREPARE_CAST_METHODS_MACRO ( Cast, typename ThisClass, M4D::Imaging::AImageRegion );

        static const unsigned Dimension = Dim;
        typedef Vector< int, Dim >			PointType;
        typedef Vector< float, Dim >		ExtentType; // typedefs for gcc4.2 error that cannot parse these templates as default parameters
        typedef Vector< unsigned, Dim >		SizeType;

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
	
        unsigned
        GetDimension() const {
                return Dimension;
        }

        unsigned
        GetSize ( unsigned dim ) const {
                return _size[dim];
        }

        Vector< unsigned, Dimension >
        GetSize() const {
                return _size;
        }

        int
        GetMinimum ( unsigned dim ) const {
                return _origin[dim];
        }

        PointType
        GetMinimum() const {
                return _origin;
        }

        int
        GetMaximum ( unsigned dim ) const {
                return _origin[dim] + _size[dim];
        }

        PointType
        GetMaximum() const {
                return _origin + PointType ( ( int32* ) _size.GetData() );
        }

        Vector< float, Dimension >
        GetElementExtents() const {
                return this->_elementExtents;
        }

        Vector< float, Dimension >
        GetRealMinimum() const {
                return VectorMemberProduct ( _origin, _elementExtents );
        }

        Vector< float, Dimension >
        GetRealMaximum() const {
                return VectorMemberProduct ( GetMaximum(), _elementExtents );
        }

        Vector< float, Dimension >
        GetRealSize() const {
                return VectorMemberProduct ( _size, _elementExtents );
        }

protected:
        AImageRegionDim (
                const SizeType	&size = SizeType ( 0 ),
                const PointType	&origin = PointType ( 0 ),
                const ExtentType	&elementExtents = ExtentType ( 1.0f )
        ) : _elementExtents ( elementExtents ), _size ( size ), _origin ( origin ) {}

        AImageRegionDim ( const AImageRegionDim &region )
                        : _elementExtents ( region._elementExtents ), _size ( region._size ), _origin ( region._origin ) {}

        Vector< float, Dimension >	_elementExtents;
        Vector< unsigned, Dimension >	_size;
        Vector< int, Dim >		_origin;

};

}/*namespace Imaging*/
/** @} */

}/*namespace M4D*/

#endif /*_AIMAGE_REGION_H*/
