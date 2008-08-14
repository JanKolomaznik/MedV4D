#ifndef _ABSTRACT_DATA_SET_H
#define _ABSTRACT_DATA_SET_H

#include <boost/shared_ptr.hpp>
#include "TimeStamp.h"
#include "Thread.h"
#include "Imaging/dataSetClassEnum.h"

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

class AbstractDataSet
{
public:
  
	/**
	 * Smart pointer to AbstractDataSet.
	 **/
	typedef boost::shared_ptr< AbstractDataSet > ADataSetPtr;

	class EAbstractDataSetCastProblem: public ErrorHandling::ExceptionCastProblem
	{
	public:
		EAbstractDataSetCastProblem() throw() 
			: ErrorHandling::ExceptionCastProblem( "Cast to AbstractDataSet impossible." ) {}
	};

	virtual
	~AbstractDataSet(){ }

	const M4D::Common::TimeStamp&
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
	static typename AbstractDataSet::ADataSetPtr
	CastDataSet( boost::shared_ptr< DatasetType > & dataset )
	{
		if( dynamic_cast< DatasetType * >( dataset.get() ) == NULL ) {
			throw EAbstractDataSetCastProblem();
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

protected:
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

	mutable ReadWriteLock	_structureLock;
private:
	DataSetType	_datasetType;

};


}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*_ABSTRACT_DATA_SET_H*/

