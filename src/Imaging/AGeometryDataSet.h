/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AGeometryDataSet.h 
 * @{ 
 **/

#ifndef _AGEOMETRY_DATA_SET_H
#define _AGEOMETRY_DATA_SET_H

#include <boost/shared_ptr.hpp>
#include "Imaging/AbstractDataSet.h"
#include "Imaging/GeometryDataSetFactory.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

class AGeometryDataSet: public AbstractDataSet
{
public:
	MANDATORY_DATASET_DEFINITIONS_THIS_MACRO( AGeometryDataSet );
	MANDATORY_DATASET_DEFINITIONS_PREDEC_MACRO( AbstractDataSet );
	PREPARE_CAST_METHODS_MACRO;
	IS_NOT_CONSTRUCTABLE_MACRO;

protected:
	AGeometryDataSet( DataSetType datasetType ): AbstractDataSet( datasetType ) 
		{}
};

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_AGEOMETRY_DATA_SET_H*/

