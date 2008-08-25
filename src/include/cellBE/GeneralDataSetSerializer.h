#ifndef GENERAL_DATASET_SERIALIZER_H
#define GENERAL_DATASET_SERIALIZER_H

#include "AbstractDataSetSerializer.h"

namespace M4D
{
namespace CellBE
{

/**
 *  General datSet serializer is some kind of recognizer of dataSet
 *  types. For given AbstractDataSet instance is able to recognize
 *  its type (based on template) and return appropriate dataSetSerializer.
 */

class GeneralDataSetSerializer
{

public:
  /**
   *  The apropriate serializer to given dataSet is returned
   *  based on GetDatasetType() returned value identifiing type
   *  of dataSet.
   */
  static AbstractDataSetSerializer *GetDataSetSerializer( 
    M4D::Imaging::AbstractDataSet *dataSet);
  
  /**
   *  Reads dataSet type and according it will use appropriate
   *  dataSetSerializer that can instantiate dataSet according
   *  recieved dataSetProperties.
   */
  static 
  void DeSerializeDataSetProperties( 
      AbstractDataSetSerializer **dataSetSerializer
      , M4D::Imaging::AbstractDataSet::ADataSetPtr *returnedDataSet
      , M4D::CellBE::NetStream &s);

};

}
}

#endif

