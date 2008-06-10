#ifndef _MODIFICATION_MANAGER_H
#define _MODIFICATION_MANAGER_H

#include "Common.h"
#include "Thread.h"
#include "TimeStamp.h"

namespace M4D
{
namespace Imaging
{

enum ModificationState{ 
	MS_DIRTY,
	MS_MODIFIED,
	MS_CANCELED
};

class ModificationBBox
{
public:

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
	WaitUntilModified();
protected:
	ModificationState	_state;
};

class ModificationBBox2D: public ModificationBBox
{

};

class ModificationBBox3D: public ModificationBBox2D
{

};

class ModificationBBox4D: public ModificationBBox3D
{

};


class ModificationManager
{
public:
	ModificationManager(){}
	
	~ModificationManager(){}

	void
	Reset();
private:
	Common::TimeStamp	_actualTimestamp;

	Common::TimeStamp	_lastStoredTimestamp;
};


}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*_MODIFICATION_MANAGER_H*/
