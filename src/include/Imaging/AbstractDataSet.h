#ifndef _ABSTRACT_DATA_SET_H
#define _ABSTRACT_DATA_SET_H

#include <boost/shared_ptr.hpp>
#include "TimeStamp.h"
#include "Imaging/dataSetProperties.h"
#include "cellBE/iPublicJob.h"

namespace M4D
{
namespace Imaging
{

class ReadWriteLock
{
public:
	ReadWriteLock():
		_canReaderAccess( true ), _readerCount( 0 ) 
		{ }

	bool 
	TryLockDataset();

	void 
	LockDataset();

	void
	UnlockDataset();

	void
	UpgradeToExclusiveLock();

	void
	DowngradeFromExclusiveLock();

	bool
	TryExclusiveLockDataset();

	void
	ExclusiveLockDataset();

	void
	ExclusiveUnlockDataset();
private:
	bool	_canReaderAccess;
	int	_readerCount;

};

class AbstractDataSet
{
public:
	/**
	 * Smart pointer to AbstractDataSet.
	 **/
	typedef boost::shared_ptr< AbstractDataSet > ADataSetPtr;

	const M4D::Common::TimeStamp&
	GetStructureTimestamp()const
		{ return _structureTimestamp; }

	/*template< typename DatasetType >
	static AbstractDataSet &
	CastDataSet(  DatasetType & dataset )
	{
		//TODO - handle exception well
		return dynamic_cast< AbstractDataSet & >( dataset );
	}

	template< typename DatasetType >
	static const AbstractDataSet &
	CastDataSet( const DatasetType & dataset )
	{
		//TODO - handle exception well
		return dynamic_cast< const AbstractDataSet & >( dataset );
	}*/

	template< typename DatasetType >
	static typename AbstractDataSet::ADataSetPtr
	CastDataSet( boost::shared_ptr< DatasetType > & dataset )
	{
		if( dynamic_cast< DatasetType * >( dataset.get() ) == NULL ) {
		//TODO throw exception
		}

		return boost::static_pointer_cast< AbstractDataSet >( dataset );
	}

	bool 
	TryLockDataset()const;

	void 
	LockDataset()const;

	void
	UnlockDataset()const;

	/**
	 * If user already has normal lock - he can ask for upgrade to Exclusive lock.
	 **/
	void
	UpgradeToExclusiveLock()const;

	/**
	 * User can downgrade exclusive lock to normal lock without worrying that
	 * someone else will get exclusive access first.
	 **/
	void
	DowngradeFromExclusiveLock()const;

	bool
	TryExclusiveLockDataset()const;

	void
	ExclusiveLockDataset()const;

	void
	ExclusiveUnlockDataset()const;

  /**
   *  Properties of dataset. Used to sending to server.
   *  This is pointer to base abstract properties class.
   *  !!! Each new type of dataSet derived from this class
   *  should declare new properties type derived from 
   *  DataSetPropertiesTemplate class (dataSetProperties.h) 
   *  with template param of type DataSetType(dataSetTypeEnums.h).
   *  This new enum type should be also added to enum with a new
   *  data set class !!!
   */
  DataSetPropertiesAbstract *_properties;

  /**
   *  Each special succesor should implement this functions in
   *  its own manner.
   */
  virtual void Serialize( M4D::CellBE::iPublicJob *job) = 0;
  virtual void DeSerialize( M4D::CellBE::iPublicJob *job) = 0;


protected:
	/**
	 * Increase structure timestamp - only helper function for successors.
	 **/
	void
	IncStructureTimestamp()
		{ ++_structureTimestamp; }

	/**
	 * Time stamp of structure changes. When internal structure is 
	 * changed (reallocation of buffers, etc.) timestamp is increased.
	 **/
	M4D::Common::TimeStamp	_structureTimestamp;

	mutable ReadWriteLock	_structureLock;
private:

};


}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*_ABSTRACT_DATA_SET_H*/

