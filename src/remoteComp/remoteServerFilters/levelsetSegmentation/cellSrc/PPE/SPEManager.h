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

#ifdef FOR_CELL
struct Tppu_pthread_data
{
	spe_context_ptr_t spe_ctx;
	pthread_t pthread;
	void *argp;
};
#endif

class SPEManager
{
public:
	SPEManager();
	~SPEManager();

	void RunSPEs(ConfigStructures *conf);
	void SendCommand(ESPUCommands &cmd);
	void WaitForCommanResult();

	uint32 GetSPECount()
	{
		return speCount;
	}

	void InitProgramProps();

	TimeStepType RunUpdateCalc();
	double ApplyUpdate(TimeStepType dt);
	void RunPropagateLayerVals();

	SPURequestsDispatcher::TWorkManager *_workManager;
	

private:
	uint32 speCount;
	
	TimeStepType MergeTimesteps();
	TimeStepType MergeRMSs();
	
	void RunDispatchers();
	
	SPURequestsDispatcher *m_requestDispatcher;

};

}
}
#endif /*SPEMANAGER_H_*/
