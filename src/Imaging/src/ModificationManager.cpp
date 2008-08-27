#include "Imaging/ModificationManager.h"

#include "Functors.h"
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
namespace Imaging
{

	
ReaderBBoxInterface::~ReaderBBoxInterface()
{ 
	delete _boundingBox; 
}

bool
ReaderBBoxInterface::IsDirty()const
{
	return GetState() == MS_DIRTY;
}

bool
ReaderBBoxInterface::IsModified()const
{
	return GetState() == MS_MODIFIED;
}

ModificationState
ReaderBBoxInterface::GetState()const
{
	return _state;
}

ModificationState
ReaderBBoxInterface::WaitWhileDirty()const
{
	while( 1 ){
		Multithreading::ScopedLock lock( _accessLock );
		if( _state == MS_DIRTY ) {
		//TODO - sleep

		} else {
			return _state;
		}
	}
}

ProxyReaderBBox::ProxyReaderBBox( Common::TimeStamp timestamp, ModificationManager* manager, ModificationBBox* boundingBox )
	: ReaderBBoxInterface( timestamp, manager, boundingBox ) 
{
	_changeIterator = _manager->ChangesReverseBegin();
}

ModificationState
ProxyReaderBBox::GetState()const
{
	while( _changeIterator != _manager->ChangesReverseEnd() 
		&& (*_changeIterator)->GetTimeStamp() >= _changeTimestamp ) 
	{
		ModificationState state = (*_changeIterator)->GetState();
		if( state != MS_MODIFIED 
			&& _boundingBox->Incident( (*_changeIterator)->GetBoundingBox() ) )
		{
			return state;
		}
		++_changeIterator;
		
	}
	return MS_MODIFIED;
}

ModificationState
ProxyReaderBBox::WaitWhileDirty()const
{
	ModificationState state = MS_MODIFIED;
	//We check if previous wait - finished in MS_MODIFIED state (not in canceled), and
	//actual bounding box has state dirty.
	while( ( state == MS_MODIFIED ) && ( ( state = GetState() ) == MS_DIRTY ) ) {
		state = (*_changeIterator)->WaitWhileDirty();
	}
	return state;
}


void
WriterBBoxInterface::SetState( ModificationState state )
{
	Multithreading::ScopedLock lock( _accessLock );

	_state = state;
}

void
WriterBBoxInterface::SetModified()
{
	SetState( MS_MODIFIED );
}

//******************************************************************************
//******************************************************************************
//******************************************************************************

ModificationBBox::ModificationBBox( 
		int32 x1, 
		int32 y1, 
		int32 x2, 
		int32 y2 
		) 
{
	_dimension = 2;
	_first = new int32[_dimension];
	_first[0] = x1;
	_first[1] = y1;

	_second = new int32[_dimension];
	_second[0] = x2;
	_second[1] = y2;
}

ModificationBBox::ModificationBBox( 
		int32 x1, 
		int32 y1, 
		int32 z1, 
		int32 x2, 
		int32 y2, 
		int32 z2 
		)
{
	_dimension = 3;
	_first = new int32[_dimension];
	_first[0] = x1;
	_first[1] = y1;
	_first[2] = z1;

	_second = new int32[_dimension];
	_second[0] = x2;
	_second[1] = y2;
	_second[2] = z2;
}

ModificationBBox::ModificationBBox( 
		int32 x1, 
		int32 y1, 
		int32 z1, 
		int32 t1, 
		int32 x2, 
		int32 y2, 
		int32 z2, 
		int32 t2 
		)
{
	_dimension = 4;
	_first = new int32[_dimension];
	_first[0] = x1;
	_first[1] = y1;
	_first[2] = z1;
	_first[3] = t1;

	_second = new int32[_dimension];
	_second[0] = x2;
	_second[1] = y2;
	_second[2] = z2;
	_second[3] = t2;
}

bool
ModificationBBox::Incident( const ModificationBBox & bbox )const
{
	//We compare only in dimensions available for both boxes
	unsigned dim = min( _dimension, bbox._dimension );
	bool result = true;

	for( unsigned d = 0; d < dim; ++d ) {
		result = result && ( min( _first[d], _second[d] ) < max( bbox._first[d], bbox._second[d] ) ) 
			&& ( max( _first[d], _second[d] ) > min( bbox._first[d], bbox._second[d] ) );
	}

	return result;
}
//******************************************************************************
//******************************************************************************
//******************************************************************************

ModificationManager::ModificationManager()
	: _actualTimestamp(), _lastStoredTimestamp( _actualTimestamp )
{

}

ModificationManager::~ModificationManager()
{
	{
		Multithreading::RecursiveScopedLock lock( _accessLock );

		std::for_each( _changes.begin(), _changes.end(), M4D::Functors::Deletor< WriterBBoxInterface *>() );
	}
}

WriterBBoxInterface &
ModificationManager::AddMod2D( 
	int32 x1, 
	int32 y1, 
	int32 x2, 
	int32 y2 
	)
{
	Multithreading::RecursiveScopedLock lock( _accessLock );

	_actualTimestamp.Increase();
	//TODO - construction of right object
	WriterBBoxInterface *change = new WriterBBoxInterface( _actualTimestamp, this, new ModificationBBox( x1, y1, x2, y2 ) );
	
	_changes.push_back( change );

	return *change;
}

ReaderBBoxInterface::Ptr
ModificationManager::GetMod2D( 
	int32 x1, 
	int32 y1, 
	int32 x2, 
	int32 y2 
	)
{
	//TODO
	Multithreading::RecursiveScopedLock lock( _accessLock );

	//TODO - construction of right object
	ReaderBBoxInterface *changeProxy = new ProxyReaderBBox( _actualTimestamp, this, new ModificationBBox( x1, y1, x2, y2 ) );

	return ReaderBBoxInterface::Ptr( changeProxy );
}

WriterBBoxInterface &
ModificationManager::AddMod3D( 
	int32 x1, 
	int32 y1, 
	int32 z1, 
	int32 x2, 
	int32 y2, 
	int32 z2 
	)
{
	Multithreading::RecursiveScopedLock lock( _accessLock );

	_actualTimestamp.Increase();
	//TODO - construction of right object
	WriterBBoxInterface *change = new WriterBBoxInterface( _actualTimestamp, this, new ModificationBBox( x1, y1, z1, x2, y2, z2 ) );
	
	_changes.push_back( change );

	return *change;
}

ReaderBBoxInterface::Ptr
ModificationManager::GetMod3D( 
	int32 x1, 
	int32 y1, 
	int32 z1, 
	int32 x2, 
	int32 y2, 
	int32 z2 
	)
{
	//TODO
	Multithreading::RecursiveScopedLock lock( _accessLock );

	//TODO - construction of right object
	ReaderBBoxInterface *changeProxy = new ProxyReaderBBox( _actualTimestamp, this, new ModificationBBox( x1, y1, z1, x2, y2, z2 ) );

	return ReaderBBoxInterface::Ptr( changeProxy );
}

//Predicate used in find_if algorithm
struct ChangeTimestampComparator
{
	ChangeTimestampComparator( const Common::TimeStamp & changeStamp ):
		_changeStamp( changeStamp ) {}
	
	bool
	operator()( WriterBBoxInterface * change )
	{ return change->GetTimeStamp() >= _changeStamp; }

	const Common::TimeStamp & _changeStamp;
};

ModificationManager::ChangeIterator 
ModificationManager::GetChangeBBox( const Common::TimeStamp & changeStamp )
{

	//TODO
	Multithreading::RecursiveScopedLock lock( _accessLock );

	return std::find_if( _changes.begin(), _changes.end(), ChangeTimestampComparator( changeStamp ) );
}

	
ModificationManager::ChangeIterator 
ModificationManager::ChangesBegin()
{
	//TODO
	Multithreading::RecursiveScopedLock lock( _accessLock );

	return _changes.begin();
}

ModificationManager::ConstChangeIterator 
ModificationManager::ChangesBegin()const
{
	//TODO
	Multithreading::RecursiveScopedLock lock( _accessLock );

	return _changes.begin();
}

ModificationManager::ChangeIterator 
ModificationManager::ChangesEnd()
{
	//TODO
	Multithreading::RecursiveScopedLock lock( _accessLock );

	return _changes.end();
}

ModificationManager::ConstChangeIterator 
ModificationManager::ChangesEnd()const
{
	//TODO
	Multithreading::RecursiveScopedLock lock( _accessLock );

	return _changes.end();
}

ModificationManager::ChangeReverseIterator 
ModificationManager::ChangesReverseBegin()
{
	//TODO
	Multithreading::RecursiveScopedLock lock( _accessLock );

	return _changes.rbegin();
}

ModificationManager::ConstChangeReverseIterator 
ModificationManager::ChangesReverseBegin()const
{
	//TODO
	Multithreading::RecursiveScopedLock lock( _accessLock );

	return _changes.rbegin();
}

ModificationManager::ChangeReverseIterator 
ModificationManager::ChangesReverseEnd()
{
	//TODO
	Multithreading::RecursiveScopedLock lock( _accessLock );

	return _changes.rend();
}

ModificationManager::ConstChangeReverseIterator 
ModificationManager::ChangesReverseEnd()const
{
	//TODO
	Multithreading::RecursiveScopedLock lock( _accessLock );

	return _changes.rend();
}


void
ModificationManager::Reset()
{
	//TODO
	Multithreading::RecursiveScopedLock lock( _accessLock );

	_lastStoredTimestamp = ++_actualTimestamp;
}


}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */