#ifndef GENERAL_DATASET_SERIALIZER_H
#define GENERAL_DATASET_SERIALIZER_H

#include "AbstractDataSetSerializer.h"

namespace M4D
{
namespace CellBE
{

/**
 *  Interface that is given to Imaging library user as an abstraction of Job.
 *  It has sending and retrival ability in scatter gather manner.
 *  Used to send and read dataSets.
 */

class GeneralDataSetSerializer
{

public:  
  
  static void SerializeDataSetProperties( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::NetStream &s);

  static void SerializeDataSet( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::iPublicJob *j);

  static M4D::Imaging::AbstractDataSet *
    DeSerializeDataSetProperties( 
      M4D::Imaging::AbstractDataSet *dataSet,
      M4D::CellBE::NetStream &s);

  static void DeSerializeDataSet( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::iPublicJob *j);

};

}
}

#endif

