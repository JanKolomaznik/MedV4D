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
  
  /**
   *  According given dataSet type uses appropriate dataSet
   *  serializer that writes all properties about given dataset needed for
   *  recreation the same instance on the other network side
   */
  static void SerializeDataSetProperties( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::NetStream &s);

  /**
   *  According given dataSet type uses appropriate dataSet
   *  serializer taht writes the whole content of given dataSet
   *  to the network through PutDataPiece methodes of iPuplicJob
   *  interface.
   */
  static void SerializeDataSet( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::iPublicJob *j);

  /**
   *  Reads dataSet type and according it will use appropriate
   *  dataSetSerializer that can instantiate dataSet according
   *  recieved dataSetProperties. The serializer is returned through
   *  second param to be used later to read actual dataSet data.
   */
  static M4D::Imaging::AbstractDataSet *
    DeSerializeDataSetProperties( 
      //AbstractDataSetSerializer **dataSetSerializer,
      M4D::CellBE::NetStream &s);

  static void DeSerializeDataSet( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::iPublicJob *j);

  static AbstractDataSetSerializer *GetDataSetSerializer( 
    M4D::Imaging::AbstractDataSet *dataSet);

};

}
}

#endif

