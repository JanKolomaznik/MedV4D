#ifndef ABSTRACT_DATASET_SERIALIZER_H
#define ABSTRACT_DATASET_SERIALIZER_H

#include "Imaging/dataSetClassEnum.h"
#include "Imaging/AbstractDataSet.h"

#include "cellBE/netStream.h"
#include "cellBE/iPublicJob.h"

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
	virtual void DeSerialize( M4D::CellBE::iPublicJob *job) = 0;

  static void SerializeProperties( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::NetStream &s);

  static void SerializeDataSet( 
    M4D::Imaging::AbstractDataSet *dataSet,
    M4D::CellBE::iPublicJob *j);

};

}
}

#endif

