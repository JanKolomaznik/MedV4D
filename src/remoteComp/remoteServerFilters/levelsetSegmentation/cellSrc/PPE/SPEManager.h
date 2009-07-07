#ifndef SPEMANAGER_H_
#define SPEMANAGER_H_

#include "SPURequestsDispatcher.h"

namespace M4D
{
namespace Cell
{

class SPEManager
{
public:
	SPEManager(WorkManager *wm);
	~SPEManager();

	static uint32 GetSPECount();
	
	void Init();
	
#ifdef FOR_CELL
	void StopSPEs();
	void StartSPEs();
#else
	void StopSims();
	void StartSims();
#endif

	TimeStepType RunUpdateCalc();
	double ApplyUpdate(TimeStepType dt);
	void RunPropagateLayerVals();

private:
	static uint32 speCount;
	
	TimeStepType MergeTimesteps();
	TimeStepType MergeRMSs();
	
	WorkManager *_workManager;
	SPURequestsDispatcher _requestDispatcher;
};

}
}
#endif /*SPEMANAGER_H_*/
