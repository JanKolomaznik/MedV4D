#include "Imaging/AbstractDataSet.h"


namespace M4D
{
namespace Imaging
{

bool 
AbstractDataSet::TryLockDataset()
{
	//TODO
	return false;
}

void
AbstractDataSet::LockDataset()
{

}

void
AbstractDataSet::UnlockDataset()
{

}

void
AbstractDataSet::UpgradeToExclusiveLock()
{

}

void
AbstractDataSet::DowngradeFromExclusiveLock()
{

}

bool
AbstractDataSet::TryExclusiveLockDataset()
{
	//TODO
	return false;
}

void
AbstractDataSet::ExclusiveLockDataset()
{

}

void
AbstractDataSet::ExclusiveUnlockDataset()
{

}

}/*namespace Imaging*/
}/*namespace M4D*/

