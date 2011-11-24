/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file ADataset.cpp
 * @{
 **/

#include "MedV4D/Imaging/ADataset.h"
#include "MedV4D/Imaging/DatasetFactory.h"
/**
 *  @addtogroup imaging Imaging Library
 *  @{
 *
 *  @author Jan Kolomaznik
 */

namespace M4D
{
namespace Imaging {

//TODO synchronization
bool
ReadWriteLock::TryLockDataset()
{
        Multithreading::ScopedLock lock ( _accessLock );

        if ( !_canReaderAccess ) {
                return false;
        }

        ++_readerCount;
        return true;
}

void
ReadWriteLock::LockDataset()
{
        while ( true ) {
                Multithreading::ScopedLock lock ( _accessLock );
                if ( !_canReaderAccess ) {
                        lock.unlock();
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
        Multithreading::ScopedLock lock ( _accessLock );

        ASSERT ( _readerCount > 0 );
        //TODO - check and exception
        --_readerCount;
}

void
ReadWriteLock::UpgradeToExclusiveLock()
{
        while ( true ) {
                Multithreading::ScopedLock lock ( _accessLock );

                ASSERT ( _readerCount > 0 );

                _canReaderAccess = false;


                if ( _readerCount > 1 ) {
                        lock.unlock();
                        //TODO wait
                } else {
                        _exclusiveLock = true;
                        return;
                }
        }
}

void
ReadWriteLock::DowngradeFromExclusiveLock()
{
        Multithreading::ScopedLock lock ( _accessLock );

        _exclusiveLock = false;

        if ( _exclusiveWaitingCount == 0 ) {
                _canReaderAccess = true;
        }
}

bool
ReadWriteLock::TryExclusiveLockDataset()
{
        Multithreading::ScopedLock lock ( _accessLock );

        if ( !_exclusiveLock && _readerCount == 0 ) {
                _canReaderAccess = false;
                _exclusiveLock = true;
                return true;
        } else {
                return false;
        }
}

void
ReadWriteLock::ExclusiveLockDataset()
{
        Multithreading::ScopedLock tmplock ( _accessLock );

        ++_exclusiveWaitingCount;

        tmplock.unlock();

        while ( true ) {
                Multithreading::ScopedLock lock ( _accessLock );

                _canReaderAccess = false;


                if ( _exclusiveLock || _readerCount > 0 ) {
                        lock.unlock();
                        //TODO wait
                } else {
                        --_exclusiveWaitingCount;
                        _exclusiveLock = true;
                        return;
                }
        }
}

void
ReadWriteLock::ExclusiveUnlockDataset()
{
        Multithreading::ScopedLock lock ( _accessLock );

        _exclusiveLock = false;

        if ( _exclusiveWaitingCount == 0 ) {
                _canReaderAccess = true;
        }

}

//******************************************************************************
//******************************************************************************

void
ADataset::Serialize ( M4D::IO::OutStream &stream )
{
        DatasetFactory::SerializeDataset ( stream, *this );
}

void
ADataset::Deserialize ( M4D::IO::InStream &stream )
{
        DatasetFactory::DeserializeDataset ( stream, *this );
}

bool
ADataset::TryLockDataset() const
{
        return _structureLock.TryLockDataset();
}

void
ADataset::LockDataset() const
{
        _structureLock.LockDataset();
}

void
ADataset::UnlockDataset() const
{
        _structureLock.UnlockDataset();
}

void
ADataset::UpgradeToExclusiveLock() const
{
        _structureLock.UpgradeToExclusiveLock();
}

void
ADataset::DowngradeFromExclusiveLock() const
{
        _structureLock.DowngradeFromExclusiveLock();
}

bool
ADataset::TryExclusiveLockDataset() const
{
        return 	_structureLock.TryExclusiveLockDataset();
}

void
ADataset::ExclusiveLockDataset() const
{
        _structureLock.ExclusiveLockDataset();
}

void
ADataset::ExclusiveUnlockDataset() const
{
        _structureLock.ExclusiveUnlockDataset();
}

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */


/** @} */

