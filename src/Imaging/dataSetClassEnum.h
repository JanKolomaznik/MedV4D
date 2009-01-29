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

enum DataSetType
{
  DATASET_IMAGE,
  DATASET_TRIANGLE_MESH, // TODO
  DATASET_SLICED_GEOMETRY
};

}//namespace M4D
}//namespace Imaging

/** @} */

#endif /*DATASET_CLASS_ENUM_H*/


/** @} */

