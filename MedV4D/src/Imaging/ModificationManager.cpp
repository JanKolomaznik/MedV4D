/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file ModificationManager.cpp
 * @{
 **/

#include "MedV4D/Imaging/ModificationManager.h"

#include "MedV4D/Common/Functors.h"
#include <algorithm>
#include <cstdlib>

using namespace std;

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 *
 *  @author Jan Kolomaznik
 */

namespace M4D
{
namespace Imaging {


ReaderBBoxInterface::~ReaderBBoxInterface()
{
        delete _boundingBox;
}

bool
ReaderBBoxInterface::IsDirty() const
{
        return GetState() == MS_DIRTY;
}

bool
ReaderBBoxInterface::IsModified() const
{
        return GetState() == MS_MODIFIED;
}

ModificationState
ReaderBBoxInterface::GetState() const
{
        Multithreading::RecursiveScopedLock lock ( _accessLock );
        return _state;
}

ModificationState
ReaderBBoxInterface::WaitWhileDirty() const
{
        while ( 1 ) {
                Multithreading::RecursiveScopedLock lock ( _accessLock );
                if ( _state == MS_DIRTY ) {
                        //TODO - sleep

                } else {
                        return _state;
                }
        }
}

ProxyReaderBBox::ProxyReaderBBox ( const Common::TimeStamp &timestamp, ModificationManager* manager, ModificationBBox* boundingBox )
                : ReaderBBoxInterface ( timestamp, manager, boundingBox )
{
        _changeIterator = _manager->ChangesReverseBegin();
}

ModificationState
ProxyReaderBBox::GetState() const
{
        while ( _changeIterator != _manager->ChangesReverseEnd()
                        && ( *_changeIterator )->GetTimeStamp() >= _changeTimestamp ) {
                ModificationState state = ( *_changeIterator )->GetState();
                if ( state != MS_MODIFIED
                                && _boundingBox->Incident ( ( *_changeIterator )->GetBoundingBox() ) ) {
                        return state;
                }
                ++_changeIterator;

        }
        return MS_MODIFIED;
}

ModificationState
ProxyReaderBBox::WaitWhileDirty() const
{
        ModificationState state = MS_MODIFIED;
        //We check if previous wait - finished in MS_MODIFIED state (not in canceled), and
        //actual bounding box has state dirty.
        while ( ( state == MS_MODIFIED ) && ( ( state = GetState() ) == MS_DIRTY ) ) {
                state = ( *_changeIterator )->WaitWhileDirty();
        }
        return state;
}


void
WriterBBoxInterface::SetState ( ModificationState state )
{
        Multithreading::RecursiveScopedLock lock ( _accessLock );

        _state = state;
}

void
WriterBBoxInterface::SetModified()
{
        SetState ( MS_MODIFIED );
}

//******************************************************************************
//******************************************************************************
//******************************************************************************



bool
ModificationBBox::Incident ( const ModificationBBox & bbox ) const
{
        //We compare only in dimensions available for both boxes
        unsigned dim = min ( _dimension, bbox._dimension );
        bool result = true;

        for ( unsigned d = 0; d < dim; ++d ) {
                result = result && ( min ( _first[d], _second[d] ) < max ( bbox._first[d], bbox._second[d] ) )
                         && ( max ( _first[d], _second[d] ) > min ( bbox._first[d], bbox._second[d] ) );
        }

        return result;
}

ModificationBBox
mergeModificationBBoxes( const ModificationBBox &aFirst, const ModificationBBox &aSecond )
{
	if ( aFirst._dimension != aSecond._dimension ) {
		_THROW_ ErrorHandling::EBadParameter( "Bounding boxes don't have same dimension" );
	}
	ModificationBBox tmp = aFirst;
	tmp.merge( aSecond );
	return tmp;
}
//******************************************************************************
//******************************************************************************
//******************************************************************************

ModificationManager::ModificationManager()
                : _actualTimestamp(), _lastStoredTimestamp ( _actualTimestamp )
{

}

ModificationManager::~ModificationManager()
{
        {
                Multithreading::RecursiveScopedLock lock ( _accessLock );

                std::for_each ( _changes.begin(), _changes.end(), M4D::Functors::Deletor< WriterBBoxInterface *>() );
        }
}


//Predicate used in find_if algorithm
struct ChangeTimestampComparator {
        ChangeTimestampComparator ( const Common::TimeStamp & changeStamp ) :
                        _changeStamp ( changeStamp ) {}

        bool
        operator() ( WriterBBoxInterface * change ) {
                return change->GetTimeStamp() >= _changeStamp;
        }

        const Common::TimeStamp & _changeStamp;
};

ModificationManager::ChangeIterator
ModificationManager::GetChangeBBox ( const Common::TimeStamp & changeStamp )
{

        //TODO
        Multithreading::RecursiveScopedLock lock ( _accessLock );

        return std::find_if ( _changes.begin(), _changes.end(), ChangeTimestampComparator ( changeStamp ) );
}

ModificationManager::ConstChangeIterator
ModificationManager::GetChangeBBox ( const Common::TimeStamp & changeStamp )const
{

        //TODO
        Multithreading::RecursiveScopedLock lock ( _accessLock );

        return std::find_if ( _changes.begin(), _changes.end(), ChangeTimestampComparator ( changeStamp ) );
}

ModificationManager::ChangeIterator
ModificationManager::ChangesBegin()
{
        //TODO
        Multithreading::RecursiveScopedLock lock ( _accessLock );

        return _changes.begin();
}

ModificationManager::ConstChangeIterator
ModificationManager::ChangesBegin() const
{
        //TODO
        Multithreading::RecursiveScopedLock lock ( _accessLock );

        return _changes.begin();
}

ModificationManager::ChangeIterator
ModificationManager::ChangesEnd()
{
        //TODO
        Multithreading::RecursiveScopedLock lock ( _accessLock );

        return _changes.end();
}

ModificationManager::ConstChangeIterator
ModificationManager::ChangesEnd() const
{
        //TODO
        Multithreading::RecursiveScopedLock lock ( _accessLock );

        return _changes.end();
}

ModificationManager::ChangeReverseIterator
ModificationManager::ChangesReverseBegin()
{
        //TODO
        Multithreading::RecursiveScopedLock lock ( _accessLock );

        return _changes.rbegin();
}

ModificationManager::ConstChangeReverseIterator
ModificationManager::ChangesReverseBegin() const
{
        //TODO
        Multithreading::RecursiveScopedLock lock ( _accessLock );

        return _changes.rbegin();
}

ModificationManager::ChangeReverseIterator
ModificationManager::ChangesReverseEnd()
{
        //TODO
        Multithreading::RecursiveScopedLock lock ( _accessLock );

        return _changes.rend();
}

ModificationManager::ConstChangeReverseIterator
ModificationManager::ChangesReverseEnd() const
{
        //TODO
        Multithreading::RecursiveScopedLock lock ( _accessLock );

        return _changes.rend();
}


void
ModificationManager::Reset()
{
        //TODO
        Multithreading::RecursiveScopedLock lock ( _accessLock );

        _lastStoredTimestamp = ++_actualTimestamp;
}

/*
template<>
WriterBBoxInterface &
ModificationManager::AddMod< 1 >(
		Vector< int32, 1 > min,
		Vector< int32, 1 > max
		)
{
	return this->AddMod1D( min[0], max[0] );
}

template<>
WriterBBoxInterface &
ModificationManager::AddMod< 2 >(
		Vector< int32, 2 > min,
		Vector< int32, 2 > max
		)
{
	return this->AddMod2D( min[0], min[1], max[0], max[1] );
}

template<>
WriterBBoxInterface &
ModificationManager::AddMod< 3 >(
		Vector< int32, 3 > min,
		Vector< int32, 3 > max
		)
{
	return this->AddMod3D( min[0], min[1], min[2], max[0], max[1], max[2] );
}

template<>
ReaderBBoxInterface::Ptr
ModificationManager::GetMod< 1 >(
		Vector< int32, 1 > min,
		Vector< int32, 1 > max
		)
{
	return this->GetMod1D( min[0], max[0] );
}

template<>
ReaderBBoxInterface::Ptr
ModificationManager::GetMod< 2 >(
		Vector< int32, 2 > min,
		Vector< int32, 2 > max
		)
{
	return this->GetMod2D( min[0], min[1], max[0], max[1] );
}

template<>
ReaderBBoxInterface::Ptr
ModificationManager::GetMod< 3 >(
		Vector< int32, 3 > min,
		Vector< int32, 3 > max
		)
{
	return this->GetMod3D( min[0], min[1], min[2], max[0], max[1], max[2] );
}*/

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */
/** @} */

