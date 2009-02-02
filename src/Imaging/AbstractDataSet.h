/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AbstractDataSet.h 
 * @{ 
 **/

#ifndef _ABSTRACT_DATA_SET_H
#define _ABSTRACT_DATA_SET_H

#include <boost/shared_ptr.hpp>
#include "TimeStamp.h"
#include "Thread.h"
#include "Imaging/dataSetClassEnum.h"
#include "Imaging/iAccessStream.h"
#include "Imaging/DatasetDefinitionTools.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

class ReadWriteLock
{
public:
	ReadWriteLock():
		_canReaderAccess( true ), _exclusiveLock( false ), _readerCount( 0 ), _exclusiveWaitingCount( 0 ) 
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
	bool	_exclusiveLock;

	int	_readerCount;
	int	_exclusiveWaitingCount;
	Multithreading::Mutex _accessLock;

};

const uint32 DUMP_START_MAGIC_NUMBER 	= 0xFEEDDEAF;
const uint32 DUMP_HEADER_END_MAGIC_NUMBER 	= 0xDEADBEAF;
const uint32 ACTUAL_FORMAT_VERSION 		= 1;

/**
 * AbstractDataSet is predecessor of all datastructures containing data. 
 * Only concept implemented by this class is read/write locking system.
 * This locking system ensure synchronization only on dataset structure 
 * (its extents, allocated buffers, etc.), not on data contained inside.
 * Read/write lock let multiple readers to obtain access and writers 
 * have to wait. If there is at least one writer waiting, no other reader 
 * is allowed to obtain lock. And when all readers finish their work, first writer
 * get exclusive access. 
 * Synchronization on data should be implemented in successors, because 
 * it differ in each type of dataset.
 * Changes in internal structure can be detected by comparing timestamps.
 * When some change in internal structure of dataset happens timestamp is 
 * increased - so if you store value from previous access you can easily detect
 * changes.
 **/
class AbstractDataSet
{
public:
	MANDATORY_ROOT_DATASET_DEFINITIONS_MACRO( AbstractDataSet );  

	class EAbstractDataSetCastProblem: public ErrorHandling::ExceptionCastProblem
	{
	public:
		EAbstractDataSetCastProblem() throw() 
			: ErrorHandling::ExceptionCastProblem( "Cast to AbstractDataSet impossible." ) {}
	};

	virtual
	~AbstractDataSet(){ }

	/**
	*  Dump data set contents
	*/
	virtual void Dump( void) = 0;

	virtual void Serialize(iAccessStream &stream) = 0;
	virtual void DeSerialize(iAccessStream &stream) = 0;

	/**
	 * \return Actual structure timestamp - it changes when internal structure is modified.
	 **/
	M4D::Common::TimeStamp
	GetStructureTimestamp()const
		{ return _structureTimestamp; }

	DataSetType
	GetDatasetType()const
		{ return _datasetType; }

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
	static typename AbstractDataSet::Ptr
	CastDataSet( boost::shared_ptr< DatasetType > & dataset )
	{
		if( dynamic_cast< DatasetType * >( dataset.get() ) == NULL ) {
			_THROW_ EAbstractDataSetCastProblem();
		}

		return boost::static_pointer_cast< AbstractDataSet >( dataset );
	}

	/**
	 * Method try to obtain read lock.
	 * \return Whether locking was successful.
	 **/
	bool 
	TryLockDataset()const;

	/**
	 * Method wait till it obtain read lock.
	 **/
	void 
	LockDataset()const;

	/**
	 * Unlock read lock - ONLY owner of the lock can UNLOCK.
	 **/
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

	/**
	 * Method try to obtain exclusive lock.
	 * \return Whether locking was successful.
	 **/
	bool
	TryExclusiveLockDataset()const;

	/**
	 * Method wait till it obtain exclusive lock.
	 **/
	void
	ExclusiveLockDataset()const;

	/**
	 * Unlock exclusive lock - ONLY owner of the lock can UNLOCK.
	 **/
	void
	ExclusiveUnlockDataset()const;

protected:
	/**
	 * Protected constructor.
	 * \param datasetType Type of dataset - this value will be returned by GetDatasetType().
	 **/
	AbstractDataSet( DataSetType datasetType ): _datasetType( datasetType ) {}


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

	/**
	 * Read/write lock used to synchronize structure modifications and reading.
	 **/
	mutable ReadWriteLock	_structureLock;
private:
	DataSetType	_datasetType;
};


}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_ABSTRACT_DATA_SET_H*/


/** @} */

