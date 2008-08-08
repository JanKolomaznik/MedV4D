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
 *  Interface that is given to Imaging library user as an abstraction of Job.
 *  It has sending and retrival ability in scatter gather manner.
 *  Used to send and read dataSets.
 */

class AbstractDataSetSerializer
{
protected:
  M4D::Imaging::AbstractDataSet *m_dataSet;

public:  
  AbstractDataSetSerializer( M4D::Imaging::AbstractDataSet *dataSet)
    : m_dataSet( dataSet) {}

  /**
   *  Identification of particular AbstractDataSet sucessor. Each new one has 
   *  return value that is added to enumeration in dataSetTypeEnums.h header.
   */
  virtual DataSetType GetID(void) = 0;

  /**
   *  Each final sucessor has to implement this functions to allow
   *  sending all properties of that particular sucessor to server.
   */
  virtual void SerializeProperties( M4D::CellBE::NetStream &s) = 0;
  virtual void DeSerializeProperties( M4D::CellBE::NetStream &s) = 0;

  	/**
	 * Properties of dataset. Used to sending to server.
	 * This is pointer to base abstract properties class.
	 * !!! Each new type of dataSet derived from this class
	 * should declare new properties type derived from 
	 * DataSetPropertiesTemplate class (dataSetProperties.h) 
	 * with template param of type DataSetType(dataSetTypeEnums.h).
	 * This new enum type should be also added to enum with a new
	 * data set class !!!
	 **/
	//DataSetPropertiesAbstract *_properties;

	/**
	 * Each special succesor should implement this functions in
	 * its own manner.
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
};

class WrongDSetException
  : public ExceptionBase
{
};

}
}

#endif

