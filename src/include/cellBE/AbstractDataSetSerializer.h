#ifndef ABSTRACT_DATASET_SERIALIZER_H
#define ABSTRACT_DATASET_SERIALIZER_H

#include "Imaging/dataSetClassEnum.h"
#include "Imaging/AbstractDataSet.h"

#include "cellBE/netStream.h"
#include "cellBE/iPublicJob.h"
#include "cellBE/dataPieceHeader.h"
#include "Common.h"

namespace M4D
{
namespace CellBE
{

/**
 *  Abstract class for all DataSet serializers. Purpouse of dataSet
 *  serializer is serialize dataSet properties and actual dataSet.
 *  Each AbstractDataSet sucessor has to have its own serializer to
 *  be able to be transmited over network.
 */
class AbstractDataSetSerializer
{
protected:
  M4D::Imaging::AbstractDataSet *m_dataSet;

  /**
   *  DataSet is state full so Reset should reset the state
   *  so the serializer could be reused.
   */
  virtual void Reset( void) = 0;

public:  
  AbstractDataSetSerializer( M4D::Imaging::AbstractDataSet *dataSet)
    : m_dataSet( dataSet) {}

  AbstractDataSetSerializer() {}

  void SetDataSet( M4D::Imaging::AbstractDataSet *dataSet)
  {
    m_dataSet = dataSet;
    Reset();
  }

  /**
   *  Identification of particular AbstractDataSet sucessor. Each new one has 
   *  return value that is added to enumeration in dataSetTypeEnums.h header.
   */
  M4D::Imaging::DataSetType GetID(void) { return m_dataSet->GetDatasetType(); }

  /**
   *  Each final sucessor has to implement this functions to allow
   *  sending all properties of that particular sucessor to server.
   *  This method must write all properties about dataSet that is
   *  in m_dataSet pointer needed for recreation the same instance
   *  (CIS - Class info serialization) on the other network side.
   *  This MUST be serialized first. Then actual dataSet properties
   *  take place (for image that is Dimension, element type, ...) =
   *  (ACS - Actual Content Serialization).
   */
  virtual void SerializeProperties( M4D::CellBE::NetStream &s) = 0;

  /**
   *  This class already have right type of serializer, so CIS
   *  has been already performed. So pnly ACS should be performed
   *  here. It returns right dataSet instance in shared_pointer.
   */
  virtual M4D::Imaging::AbstractDataSet::ADataSetPtr
    DeSerializeProperties( M4D::CellBE::NetStream &s) = 0;

	/**
	 *  This method should preform actual dataSet data serialization
   *  through iPublicJob::PutDataPiece method.
	 **/
	virtual void Serialize( M4D::CellBE::iPublicJob *job) = 0;

  /**
   *  This method is called when dataPieceHeader arrive.
   *  The bufs should be filled with buffer where incomming data
   *  that corresponds with the header should be placed.
   */
  virtual void OnDataPieceReadRequest( DataPieceHeader *header, DataBuffs &bufs) = 0;

  /**
   *  This is called when no more data is going to be arrive (special ending
   *  header is recieved).
   */
  virtual void OnDataSetEndRead( void) = 0;

  virtual void DumpDataSet( void) {}
};

///////////////////////////////////////////

class WrongDSetException
  : public ExceptionBase
{
};

}
}

#endif

