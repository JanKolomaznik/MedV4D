#ifndef _MODIFICATION_MANAGER_H
#define _MODIFICATION_MANAGER_H

#include "Common.h"
#include "Thread.h"
#include "TimeStamp.h"
#include <boost/shared_ptr.hpp>
#include <list>

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


	ReaderBBoxInterface( Common::TimeStamp timestamp, ModificationManager* manager, ModificationBBox* boundingBox )
		: _changeTimestamp( timestamp ),  _manager( manager ), _boundingBox( boundingBox ) {}

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
	WaitWhileDirty();

protected:
	Common::TimeStamp _changeTimestamp;

	ModificationState	_state;

	ModificationManager	*_manager;

	ModificationBBox	*_boundingBox;
	
	mutable Multithreading::Mutex	_accessLock;
};

class WriterBBoxInterface : public ReaderBBoxInterface
{
public:
	WriterBBoxInterface( Common::TimeStamp timestamp, ModificationManager* manager, ModificationBBox* boundingBox )
		: ReaderBBoxInterface( timestamp, manager, boundingBox ) {}

	void
	SetState( ModificationState state );

	void
	SetModified();


};


class ModificationBBox
{
public:
	ModificationBBox( 
		int32 x1, 
		int32 y1, 
		int32 x2, 
		int32 y2 
		);

	ModificationBBox( 
		int32 x1, 
		int32 y1, 
		int32 z1, 
		int32 x2, 
		int32 y2, 
		int32 z2 
		);

	ModificationBBox( 
		int32 x1, 
		int32 y1, 
		int32 z1, 
		int32 t1, 
		int32 x2, 
		int32 y2, 
		int32 z2, 
		int32 t2 
		);

	~ModificationBBox() { delete [] _first;  delete [] _second; }

	void
	GetInterval( unsigned dim, int32 &first, int32 &second )
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

	ModificationManager();
	
	~ModificationManager();

	//ModBBoxWholeDataset&
	//GetWholeDatasetBBox();
	WriterBBoxInterface &
	AddMod2D( 
		size_t x1, 
		size_t y1, 
		size_t x2, 
		size_t y2 
		);
	
	ReaderBBoxInterface::Ptr
	GetMod2D( 
		size_t x1, 
		size_t y1, 
		size_t x2, 
		size_t y2 
		);

	WriterBBoxInterface &
	AddMod3D( 
		size_t x1, 
		size_t y1, 
		size_t z1, 
		size_t x2, 
		size_t y2, 
		size_t z2 
		);
	
	ReaderBBoxInterface::Ptr
	GetMod3D( 
		size_t x1, 
		size_t y1, 
		size_t z1, 
		size_t x2, 
		size_t y2, 
		size_t z2 
		);

	WriterBBoxInterface &
	AddMod4D( 
		size_t x1, 
		size_t y1, 
		size_t z1, 
		size_t x2, 
		size_t y2, 
		size_t z2 
		);
	
	ReaderBBoxInterface::Ptr
	GetMod4D( 
		size_t x1, 
		size_t y1, 
		size_t z1, 
		size_t t1, 
		size_t x2, 
		size_t y2, 
		size_t z2, 
		size_t t2 
		);

	ChangeIterator 
	GetChangeBBox( const Common::TimeStamp & changeStamp );

	ChangeIterator 
	ChangesBegin();

	ChangeIterator 
	ChangesEnd();

	ChangeReverseIterator 
	ChangesReverseBegin();

	ChangeReverseIterator 
	ChangesReverseEnd();

	void
	Reset();
private:
	Common::TimeStamp	_actualTimestamp;

	Common::TimeStamp	_lastStoredTimestamp;

	ChangeQueue		_changes;

	Multithreading::Mutex	_accessLock;
};

class ProxyReaderBBox: public ReaderBBoxInterface
{
public:
	ProxyReaderBBox( Common::TimeStamp timestamp, ModificationManager* manager, ModificationBBox* boundingBox );

	ModificationState
	GetState()const;

	ModificationState
	WaitWhileDirty();
protected:
	ModificationManager::ChangeReverseIterator _changeIterator;
};


}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*_MODIFICATION_MANAGER_H*/
