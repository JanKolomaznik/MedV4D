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

class ModificationManager;

class ReaderBBoxInterface
{
public:
	typedef boost::shared_ptr< ReaderBBoxInterface > Ptr;


	ReaderBBoxInterface( Common::TimeStamp timestamp, ModificationManager* manager )
		: _changeTimestamp( timestamp ),  _manager( manager ) {}

	virtual
	~ReaderBBoxInterface() {}

	bool
	IsDirty()const;

	bool
	IsModified()const;

	ModificationState
	GetState()const;

	const Common::TimeStamp &
	GetTimeStamp()const
		{ return _changeTimestamp; }

	bool
	WaitWhileDirty();

protected:
	Common::TimeStamp _changeTimestamp;

	ModificationState	_state;

	ModificationManager	*_manager;
};

class WriterBBoxInterface : public ReaderBBoxInterface
{
public:
	WriterBBoxInterface( Common::TimeStamp timestamp, ModificationManager* manager ): ReaderBBoxInterface( timestamp, manager ) {}

	void
	SetState( ModificationState );

	void
	SetModified();


};


class ModificationBBox
{
public:

	virtual 
	~ModificationBBox() {}
protected:
	ModificationBBox( unsigned dim, int *first, int *second )
		:_dimension( dim ), _first( first ), _second( second ) {}

	unsigned	_dimension;

	int 		*_first;
	int 		*_second;
};

class BBox2D: public ModificationBBox
{
public:

};

class BBox3D: public BBox2D
{
public:

};

class BBox4D: public BBox3D
{
public:

};


class ModificationManager
{
public:
	typedef std::list< WriterBBoxInterface * > 	ChangeQueue;
	typedef ChangeQueue::iterator			ChangeIterator;

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

	ChangeIterator 
	GetChangeBBox( const Common::TimeStamp & changeStamp );

	ChangeIterator 
	ChangesBegin();

	ChangeIterator 
	ChangesEnd();

	void
	Reset();
private:
	Common::TimeStamp	_actualTimestamp;

	Common::TimeStamp	_lastStoredTimestamp;

	ChangeQueue		_changes;

	Multithreading::Mutex	_accessLock;
};


}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*_MODIFICATION_MANAGER_H*/
