#ifndef SPUREQUESTSDISPATCHER_H_
#define SPUREQUESTSDISPATCHER_H_

#include "common/Thread.h"

//to remove
#include "../supportClasses.h"

#include "itkIndex.h"
#include <queue>

#include "workManager.h"

namespace M4D {
namespace Cell {

class SPURequestsDispatcher
{
public:
	void DispatchMessage(uint32 message);
//	void SendResponse(uint32 res);
	
	void DispatchPushNodeMess(uint32 message);
	void DispatchUnlinkMessage(uint32 message);
	
	typedef itk::Index<DIM> TIndex;
	typedef std::queue<uint32> TMessageQueue;
	typedef WorkManager<TIndex, float32> TWorkManager;
	typedef itk::SparseFieldLevelSetNode<TIndex> LayerNodeType;
	
//#define MAX_QUEUE_LEN 4
	TMessageQueue messageQueue;
	
	void MyPushMessage(uint32);
	uint32 MyPopMessage();
	
	M4D::Multithreading::Mutex mutex;	
	M4D::Multithreading::Mutex mutexRun;

	TWorkManager *_workManager;
	
	TimeStepType _result;
	uint32 _segmentID;
};

}
}
#endif /*SPUREQUESTSDISPATCHER_H_*/
