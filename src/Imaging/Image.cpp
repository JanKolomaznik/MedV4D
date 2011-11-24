/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file Image.cpp
 * @{
 **/

#include "MedV4D/Imaging/Image.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 *
 *  @author Jan Kolomaznik
 */

namespace M4D
{
namespace Imaging {

AImage::AImage ( uint16 dim, DimensionExtents *dimExtents )
                : ADataset ( DATASET_IMAGE ), _dimCount ( dim ), _dimensionExtents ( dimExtents )
{

}

AImage::~AImage()
{

}

const DimensionExtents &
AImage::GetDimensionExtents ( unsigned dimension ) const
{
        if ( dimension >= _dimCount ) {
                _THROW_ EBadDimension();
        }
        return _dimensionExtents[ dimension ];
}


}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */
/** @} */

