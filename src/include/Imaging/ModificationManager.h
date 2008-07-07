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


	ReaderBBoxInterface( Common::TimeStamp timestamp ): _changeTimestamp( timestamp ) {}

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
};

class WriterBBoxInterface : public ReaderBBoxInterface
{
public:
	WriterBBoxInterface( Common::TimeStamp timestamp ): ReaderBBoxInterface( timestamp ) {}

	void
	SetState( ModificationState );

	void
	SetModified();


};


class ModificationBBox
{
public:
	ModificationBBox( ModificationManager* manager )
		: _manager( manager ) { }

	virtual
	~ModificationBBox() {}

	bool
	IsDirty()const;

	bool
	IsModified()const;

	ModificationState
	GetState()const;

	const Common::TimeStamp &
	GetTimeStamp()const;

	void
	SetState( ModificationState );

	void
	SetModified();

	bool
	WaitUntilDirty();
protected:
	ModificationState	_state;

	ModificationManager	*_manager;
};

class ModBBox2D: public ModificationBBox
{
public:
	ModBBox2D( ModificationManager* manager ) 
		: ModificationBBox( manager ) { }

};

class ModBBox3D: public ModBBox2D
{
public:
	ModBBox3D( ModificationManager* manager ) 
		: ModBBox2D( manager ) { }

};

class ModBBox4D: public ModBBox3D
{
public:
	ModBBox4D( ModificationManager* manager ) 
		: ModBBox3D( manager ) { }

};


/*
class ModBBoxWholeDataset: public ModificationBBox
{
public:
	ModBBoxWholeDataset( ModificationManager* manager ) 
		: ModificationBBox( manager ) { }
	void
	ReadLock();

	void
	ReadUnlock();

	/ *void
	ReadTryLock();* /
};
*/

/*class ModBBox3D
{

};
*/
class ReadBBox3D
{

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
