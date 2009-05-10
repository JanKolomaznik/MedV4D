#ifndef SPEMANAGER_H_
#define SPEMANAGER_H_

#include "SPURequestsDispatcher.h"

#ifdef FOR_CELL
#include <libspe2.h>
#else
//#include "SPEProgramSimulator.h"
#endif

namespace M4D
{
namespace Cell
{

class SPEManager
{
public:
	SPEManager(SPURequestsDispatcher::TWorkManager *wm);
	~SPEManager();

	static uint32 GetSPECount();
	
#ifdef FOR_CELL
	void StopSPEs();
	void StartSPEs();
#else
	void StopSims();
	void StartSims();
#endif

	//void InitProgramProps();

	TimeStepType RunUpdateCalc();
	double ApplyUpdate(TimeStepType dt);
	void RunPropagateLayerVals();

	SPURequestsDispatcher::TWorkManager *_workManager;

private:
	static uint32 speCount;
	
	void UnblockDispatchers();
	
	TimeStepType MergeTimesteps();
	TimeStepType MergeRMSs();
	
//	void RunDispatchers();
	
	SPURequestsDispatcher _requestDispatcher;
};

}
}
#endif /*SPEMANAGER_H_*/
