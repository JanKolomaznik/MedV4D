#include "Imaging/AbstractDataSet.h"


namespace M4D
{
namespace Imaging
{

	//TODO synchronization
bool 
ReadWriteLock::TryLockDataset()
{
	if( !_canReaderAccess ) { 
		return false;
	}

	++_readerCount;
	return true;
}

void 
ReadWriteLock::LockDataset()
{
	while( true ) {
		if( !_canReaderAccess ) {
		//TODO wait		
		} else {
			++_readerCount;
			return;
		}
	} 
}

void
ReadWriteLock::UnlockDataset()
{
	//TODO - check and exception
	--_readerCount;
}

void
ReadWriteLock::UpgradeToExclusiveLock()
{

}

void
ReadWriteLock::DowngradeFromExclusiveLock()
{

}

bool
ReadWriteLock::TryExclusiveLockDataset()
{
	return false;
}

void
ReadWriteLock::ExclusiveLockDataset()
{

}

void
ReadWriteLock::ExclusiveUnlockDataset()
{

}

//******************************************************************************
//******************************************************************************




bool 
AbstractDataSet::TryLockDataset()const
{
	return _structureLock.TryLockDataset();
}

void
AbstractDataSet::LockDataset()const
{
	_structureLock.LockDataset();
}

void
AbstractDataSet::UnlockDataset()const
{
	_structureLock.UnlockDataset();
}

void
AbstractDataSet::UpgradeToExclusiveLock()const
{
	_structureLock.UpgradeToExclusiveLock();
}

void
AbstractDataSet::DowngradeFromExclusiveLock()const
{
	_structureLock.DowngradeFromExclusiveLock();
}

bool
AbstractDataSet::TryExclusiveLockDataset()const
{
	return 	_structureLock.TryExclusiveLockDataset();
}

void
AbstractDataSet::ExclusiveLockDataset()const
{
	_structureLock.ExclusiveLockDataset();
}

void
AbstractDataSet::ExclusiveUnlockDataset()const
{
	_structureLock.ExclusiveUnlockDataset();
}

}/*namespace Imaging*/
}/*namespace M4D*/

