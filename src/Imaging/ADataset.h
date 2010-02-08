/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ADataset.h 
 * @{ 
 **/

#ifndef _ABSTRACT_DATA_SET_H
#define _ABSTRACT_DATA_SET_H

#include <boost/shared_ptr.hpp>
#include "common/TimeStamp.h"
#include "common/Thread.h"
#include "Imaging/DatasetClassEnum.h"
#include "common/IOStreams.h"
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



/**
 * ADataset is predecessor of all datastructures containing data. 
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
class ADataset
{
public:
	MANDATORY_ROOT_DATASET_DEFINITIONS_MACRO( ADataset );  
	PREPARE_CAST_METHODS_MACRO;
	IS_NOT_CONSTRUCTABLE_MACRO;

	class EADatasetCastProblem: public ErrorHandling::ExceptionCastProblem
	{
	public:
		EADatasetCastProblem() throw() 
			: ErrorHandling::ExceptionCastProblem( "Cast to ADataset impossible." ) {}
	};

	virtual
	~ADataset(){ }

	/**
	*  Dump data set contents
	*/
	virtual void Dump( std::ostream &s) const
		{ _THROW_ ErrorHandling::ETODO(); }

	void Serialize(M4D::IO::OutStream &stream);

	void Deserialize(M4D::IO::InStream &stream);

	// interface for serialization
	/*virtual void SerializeData(M4D::IO::OutStream &stream) = 0;
	virtual void SerializeClassInfo(M4D::IO::OutStream &stream) = 0;
	virtual void SerializeProperties(M4D::IO::OutStream &stream) = 0;
	virtual void DeSerializeData(M4D::IO::InStream &stream) = 0;*/
	//virtual void DeSerializeProperties(M4D::IO::InStream &stream) = 0;

	/**
	 * \return Actual structure timestamp - it changes when internal structure is modified.
	 **/
	M4D::Common::TimeStamp
	GetStructureTimestamp()const
		{ return _structureTimestamp; }

	DatasetType
	GetDatasetType()const
		{ return _datasetType; }

	/*template< typename DatasetType >
	static ADataset &
	CastDataset(  DatasetType & dataset )
	{
		//TODO - handle exception well
		return dynamic_cast< ADataset & >( dataset );
	}

	template< typename DatasetType >
	static const ADataset &
	CastDataset( const DatasetType & dataset )
	{
		//TODO - handle exception well
		return dynamic_cast< const ADataset & >( dataset );
	}*/

	template< typename DatasetType >
	static typename ADataset::Ptr
	CastDataset( boost::shared_ptr< DatasetType > & dataset )
	{
		if( dynamic_cast< DatasetType * >( dataset.get() ) == NULL ) {
			_THROW_ EADatasetCastProblem();
		}

		return boost::static_pointer_cast< ADataset >( dataset );
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
	ADataset( DatasetType datasetType ): _datasetType( datasetType ) {}


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
	DatasetType	_datasetType;
};


}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_ABSTRACT_DATA_SET_H*/


/** @} */

