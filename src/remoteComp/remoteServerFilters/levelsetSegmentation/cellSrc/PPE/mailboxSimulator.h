#ifndef MAILBOXSIMULATOR_H_
#define MAILBOXSIMULATOR_H_

#include <queue>
#include "MedV4D/Common/Thread.h"

namespace M4D
{
namespace Cell
{

using namespace M4D::Multithreading;


class MailboxSimulator
{
public:	
	void PPEPush(uint32 mess)
	{
		ScopedLock lock(toSPEQMutex);
		toSPEQueue.push(mess);
		toSPEQCvar.notify_all();
	}
	uint32 PPEPop(void)
	{
		ScopedLock lock(fromSPEQMutex);
		uint32 val = fromSPEQueue.front();
		fromSPEQueue.pop();
		return val;
	}
	
	void SPEPush(uint32 mess)
	{
		fromSPEQueue.push(mess);
	}
	uint32 SPEPop(void)
	{
		ScopedLock lock(toSPEQMutex);
		while(toSPEQueue.empty())
			toSPEQCvar.wait(lock);
		uint32 val = toSPEQueue.front();
		toSPEQueue.pop();
		return val;
	}
	
	bool spe_queue_status()
	{
		ScopedLock lock(fromSPEQMutex);
		return ! fromSPEQueue.empty();
	}
	
	Mutex fromSPEQMutex;
	Mutex toSPEQMutex;
	
private:
	typedef std::queue<uint32> TMessageQueue;
	
	//#define MAX_QUEUE_LEN 4
	TMessageQueue fromSPEQueue;
	TMessageQueue toSPEQueue;

	
	CondVar toSPEQCvar;
};

}
}
#endif /*MAILBOXSIMULATOR_H_*/
