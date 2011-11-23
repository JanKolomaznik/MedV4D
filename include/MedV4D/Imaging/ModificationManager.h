/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ModificationManager.h 
 * @{ 
 **/

#ifndef _MODIFICATION_MANAGER_H
#define _MODIFICATION_MANAGER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/Thread.h"
#include "MedV4D/Common/TimeStamp.h"
#include <boost/shared_ptr.hpp>
#include <list>
#include "MedV4D/Common/Vector.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

enum ModificationState{ 
	MS_DIRTY,
	MS_MODIFIED,
	MS_CANCELED
};

class ModificationBBox;
class ModificationManager;

class ReaderBBoxInterface
{
public:
	typedef boost::shared_ptr< ReaderBBoxInterface > Ptr;


	ReaderBBoxInterface( const Common::TimeStamp &timestamp, ModificationManager* manager, ModificationBBox* boundingBox )
		: _changeTimestamp( timestamp ),  _state( MS_MODIFIED ), _manager( manager ), _boundingBox( boundingBox ) {}

	virtual
	~ReaderBBoxInterface();

	bool
	IsDirty()const;

	bool
	IsModified()const;

	virtual ModificationState
	GetState()const;

	Common::TimeStamp
	GetTimeStamp()const
		{ return _changeTimestamp; }

	const ModificationBBox &
	GetBoundingBox()const
		{ return *_boundingBox; }

	virtual ModificationState
	WaitWhileDirty()const;

protected:
	Common::TimeStamp _changeTimestamp;

	ModificationState	_state;

	ModificationManager	*_manager;

	ModificationBBox	*_boundingBox;
	
	mutable Multithreading::RecursiveMutex	_accessLock;
};

class WriterBBoxInterface : public ReaderBBoxInterface
{
public:
	WriterBBoxInterface( const Common::TimeStamp &timestamp, ModificationManager* manager, ModificationBBox* boundingBox )
		: ReaderBBoxInterface( timestamp, manager, boundingBox ) 
		{ _state = MS_DIRTY; }

	void
	SetState( ModificationState state );

	void
	SetModified();


};


class ModificationBBox
{
public:
	
	template< unsigned Dim >
	explicit ModificationBBox( const Vector< int32, Dim > &min, const Vector< int32, Dim > &max );

	~ModificationBBox() { delete [] _first;  delete [] _second; }

	void
	GetInterval( unsigned dim, int32 &first, int32 &second )const
		{ 
			if( dim >= _dimension )	{ 
				/*TODO exception*/ 
			} 

			first = _first[ dim ]; second = _second[ dim ]; 
		}

	bool
	Incident( const ModificationBBox & bbox )const;

protected:
	ModificationBBox( unsigned dim, int *first, int *second )
		:_dimension( dim ), _first( first ), _second( second ) {}

	unsigned	_dimension;

	int32 		*_first;
	int32 		*_second;
};

class ModificationManager
{
public:
	typedef std::list< WriterBBoxInterface * > 	ChangeQueue;
	typedef ChangeQueue::iterator			ChangeIterator;
	typedef ChangeQueue::reverse_iterator		ChangeReverseIterator;
	typedef ChangeQueue::const_iterator		ConstChangeIterator;
	typedef ChangeQueue::const_reverse_iterator	ConstChangeReverseIterator;

	ModificationManager();
	
	~ModificationManager();


	template< unsigned Dim >
	WriterBBoxInterface &
	AddMod(
		const Vector< int32, Dim > &min,
		const Vector< int32, Dim > &max
	      );

	template< unsigned Dim >
	ReaderBBoxInterface::Ptr
	GetMod(
		const Vector< int32, Dim > &min,
		const Vector< int32, Dim > &max
	      );


	ChangeIterator 
	GetChangeBBox( const Common::TimeStamp & changeStamp );

	ChangeIterator 
	ChangesBegin();

	ConstChangeIterator 
	ChangesBegin()const;

	ChangeIterator 
	ChangesEnd();

	ConstChangeIterator 
	ChangesEnd()const;

	ChangeReverseIterator 
	ChangesReverseBegin();

	ConstChangeReverseIterator 
	ChangesReverseBegin()const;

	ChangeReverseIterator 
	ChangesReverseEnd();

	ConstChangeReverseIterator 
	ChangesReverseEnd()const;

	Common::TimeStamp
	GetLastStoredTimestamp()const
		{ return _lastStoredTimestamp; }

	Common::TimeStamp
	GetActualTimestamp()const
		{ return _actualTimestamp; }

	void
	Reset();
private:
	Common::TimeStamp	_actualTimestamp;

	Common::TimeStamp	_lastStoredTimestamp;

	ChangeQueue		_changes;

	mutable Multithreading::RecursiveMutex	_accessLock;
};

class ProxyReaderBBox: public ReaderBBoxInterface
{
public:
	ProxyReaderBBox( const Common::TimeStamp &timestamp, ModificationManager* manager, ModificationBBox* boundingBox );

	ModificationState
	GetState()const;

	ModificationState
	WaitWhileDirty()const;
protected:
	mutable ModificationManager::ChangeReverseIterator _changeIterator;
};


template< unsigned Dim >
ModificationBBox::ModificationBBox( const Vector< int32, Dim > &min, const Vector< int32, Dim > &max )
{
	_dimension = Dim;
	_first = new int32[Dim];
	_second = new int32[Dim];

	for( unsigned i = 0; i < Dim; ++i ) {
		_first[i] = min[i];
		_second[i] = max[i];
	}
}

template< unsigned Dim >
WriterBBoxInterface &
ModificationManager::AddMod( 
		const Vector< int32, Dim > &min,
		const Vector< int32, Dim > &max
		)
{
	Multithreading::RecursiveScopedLock lock( _accessLock );

	_actualTimestamp.Increase();
	//TODO - construction of right object
	ModificationBBox * bbox = new ModificationBBox( min, max );
	WriterBBoxInterface *change = new WriterBBoxInterface( _actualTimestamp, this, bbox );
	
	_changes.push_back( change );

	return *change;
}

template< unsigned Dim >
ReaderBBoxInterface::Ptr
ModificationManager::GetMod( 
		const Vector< int32, Dim > &min,
		const Vector< int32, Dim > &max
		)
{
	Multithreading::RecursiveScopedLock lock( _accessLock );

	//TODO - construction of right object
	ModificationBBox * bbox = new ModificationBBox( min, max );
	ReaderBBoxInterface *changeProxy = new ProxyReaderBBox( _actualTimestamp, this, bbox );

	return ReaderBBoxInterface::Ptr( changeProxy );
}

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_MODIFICATION_MANAGER_H*/

/** @} */

