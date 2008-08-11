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
   */
  static AbstractDataSetSerializer *GetDataSetSerializer( 
    M4D::Imaging::AbstractDataSet *dataSet);
  
  /**
   *  According given dataSet type uses appropriate dataSet
   *  serializer that writes all properties about given dataset needed for
   *  recreation the same instance on the other network side.
   *  Uses GetDataSetSerializer function (see above) internaly.
   */
  static void SerializeDataSetProperties( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::NetStream &s);

  /**
   *  According given dataSet type uses appropriate dataSet
   *  serializer taht writes the whole content of given dataSet
   *  to the network through PutDataPiece methodes of iPuplicJob
   *  interface.
   *  Uses GetDataSetSerializer function (see above) internaly.
   */
  static void SerializeDataSet( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::iPublicJob *j);

  /**
   *  Reads dataSet type and according it will use appropriate
   *  dataSetSerializer that can instantiate dataSet according
   *  recieved dataSetProperties.
   */
  static M4D::Imaging::AbstractDataSet *
    DeSerializeDataSetProperties( 
      //AbstractDataSetSerializer **dataSetSerializer,
      M4D::CellBE::NetStream &s);

  //static void DeSerializeDataSet( 
  //  M4D::Imaging::AbstractDataSet *dataSet,
  //  M4D::CellBE::iPublicJob *j);

  
};

}
}

#endif

