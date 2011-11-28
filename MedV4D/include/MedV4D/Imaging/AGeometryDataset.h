/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file AGeometryDataset.h
 * @{
 **/

#ifndef _AGEOMETRY_DATA_SET_H
#define _AGEOMETRY_DATA_SET_H

#include <boost/shared_ptr.hpp>
#include "MedV4D/Imaging/ADataset.h"
#include "MedV4D/Imaging/GeometryDatasetFactory.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging {

class AGeometryDataset: public ADataset
{
public:
        MANDATORY_DATASET_DEFINITIONS_THIS_MACRO ( AGeometryDataset );
        MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO ( ADataset );
        PREPARE_CAST_METHODS_MACRO;
        IS_NOT_CONSTRUCTABLE_MACRO;

protected:
        AGeometryDataset ( DatasetType datasetType ) : ADataset ( datasetType ) {}
};

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */
/** @} */

#endif /*_AGEOMETRY_DATA_SET_H*/

