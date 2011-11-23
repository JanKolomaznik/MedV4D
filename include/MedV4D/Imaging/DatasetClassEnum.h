/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file dataSetClassEnum.h 
 * @{ 
 **/

#ifndef DATASET_CLASS_ENUM_H
#define DATASET_CLASS_ENUM_H

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

enum DatasetType
{
  DATASET_IMAGE = 10,
  DATASET_TRIANGLE_MESH = 15, // TODO
  DATASET_SLICED_GEOMETRY = 20
};

}//namespace M4D
}//namespace Imaging

/** @} */

#endif /*DATASET_CLASS_ENUM_H*/


/** @} */

