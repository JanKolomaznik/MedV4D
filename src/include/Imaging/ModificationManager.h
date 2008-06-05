#ifndef _MODIFICATION_MANAGER_H
#define _MODIFICATION_MANAGER_H

#include "Common.h"
#include "Thread.h"
#include "TimeStamp.h"

namespace M4D
{
namespace Imaging
{

struct ModificationRecordRectangle
{
	Common::TimeStamp	_modificationTime;
};

struct ModificationRecordSlice
{
	Common::TimeStamp	_modificationTime;
};

struct ModificationRecordBox
{
	Common::TimeStamp	_modificationTime;
};

class ModificationManager
{
public:
	void
	Reset();
private:
	Common::TimeStamp	_actualTimestamp;

	Common::TimeStamp	_lastStoredTimestamp;
};


}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*_MODIFICATION_MANAGER_H*/
