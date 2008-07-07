#include "Imaging/ModificationManager.h"

#include "Functors.h"
#include <algorithm>

namespace M4D
{
namespace Imaging
{

ModificationManager::ModificationManager()
	: _actualTimestamp(), _lastStoredTimestamp( _actualTimestamp )
{

}

ModificationManager::~ModificationManager()
{
	{
		Multithreading::ScopedLock lock( _accessLock );

		std::for_each( _changes.begin(), _changes.end(), M4D::Functors::Deletor< WriterBBoxInterface *>() );
	}
}

WriterBBoxInterface &
ModificationManager::AddMod2D( 
	size_t x1, 
	size_t y1, 
	size_t x2, 
	size_t y2 
	)
{
	Multithreading::ScopedLock lock( _accessLock );

	_actualTimestamp.Increase();
	//TODO - construction of right object
	WriterBBoxInterface *change = new WriterBBoxInterface( _actualTimestamp );
	
	_changes.push_back( change );

	return *change;
}

ReaderBBoxInterface::Ptr
ModificationManager::GetMod2D( 
	size_t x1, 
	size_t y1, 
	size_t x2, 
	size_t y2 
	)
{
	//TODO
	Multithreading::ScopedLock lock( _accessLock );

	//TODO - construction of right object
	ReaderBBoxInterface *changeProxy = new ReaderBBoxInterface( _actualTimestamp );

	return ReaderBBoxInterface::Ptr( changeProxy );
}

WriterBBoxInterface &
ModificationManager::AddMod3D( 
	size_t x1, 
	size_t y1, 
	size_t z1, 
	size_t x2, 
	size_t y2, 
	size_t z2 
	)
{
	Multithreading::ScopedLock lock( _accessLock );

	_actualTimestamp.Increase();
	//TODO - construction of right object
	WriterBBoxInterface *change = new WriterBBoxInterface( _actualTimestamp );
	
	_changes.push_back( change );

	return *change;
}

ReaderBBoxInterface::Ptr
ModificationManager::GetMod3D( 
	size_t x1, 
	size_t y1, 
	size_t z1, 
	size_t x2, 
	size_t y2, 
	size_t z2 
	)
{
	//TODO
	Multithreading::ScopedLock lock( _accessLock );

	//TODO - construction of right object
	ReaderBBoxInterface *changeProxy = new ReaderBBoxInterface( _actualTimestamp );

	return ReaderBBoxInterface::Ptr( changeProxy );
}

ModificationManager::ChangeIterator 
ModificationManager::GetChangeBBox( const Common::TimeStamp & changeStamp )
{
	//Predicate used in find_if algorithm
	struct ChangeTimestampComparator
	{
		ChangeTimestampComparator(  const Common::TimeStamp & changeStamp ):
			_changeStamp( changeStamp ) {}
		
		bool
		operator()( WriterBBoxInterface * change )
		{ return change->GetTimeStamp() >= _changeStamp; }

		const Common::TimeStamp & _changeStamp;
	};

	//TODO
	Multithreading::ScopedLock lock( _accessLock );

	return std::find_if( _changes.begin(), _changes.end(), ChangeTimestampComparator( changeStamp ) );
}

	
ModificationManager::ChangeIterator 
ModificationManager::ChangesBegin()
{
	//TODO
	Multithreading::ScopedLock lock( _accessLock );

	return _changes.begin();
}

ModificationManager::ChangeIterator 
ModificationManager::ChangesEnd()
{
	//TODO
	Multithreading::ScopedLock lock( _accessLock );

	return _changes.end();
}

void
ModificationManager::Reset()
{
	//TODO
	Multithreading::ScopedLock lock( _accessLock );

	_lastStoredTimestamp = ++_actualTimestamp;
}


}/*namespace Imaging*/
}/*namespace M4D*/
