#ifndef SPEMANAGER_H_
#define SPEMANAGER_H_

#include <pthread.h>

#include "SPURequestsDispatcher.h"

#ifdef FOR_CELL
#include <libspe2.h>
#else
#include "SPEProgramSimulator.h"
#endif

namespace M4D {
namespace Cell {

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
	
	uint32 GetSPECount() { return speCount; }
	
	void InitProgramProps();
	
	TimeStepType RunUpdateCalc()
	  {
		_workManager->AllocateUpdateBuffers();
		_workManager->InitCalculateChangeAndUpdActiveLayerConf();
	  	_SPEProgSim.updateSolver.UpdateFunctionProperties();
	  	
	  	TimeStepType dt = _SPEProgSim.updateSolver.CalculateChange();
	  	
	  	return dt;
	  }
	  double ApplyUpdate(TimeStepType dt)
	  {
		  _workManager->InitCalculateChangeAndUpdActiveLayerConf();
		  _workManager->InitPropagateValuesConf();
			
		  return _SPEProgSim.applyUpdateCalc.ApplyUpdate(dt);
	  }
	void RunPropagateLayerVals()
	{
		_workManager->InitPropagateValuesConf();
		_SPEProgSim.applyUpdateCalc.PropagateAllLayerValues();
	}
	
	SPURequestsDispatcher::TWorkManager *_workManager;
		
private:
	uint32 speCount;
	
#ifdef FOR_CELL
	Tppu_pthread_data *data;
#endif
	
	
	
	 SPUProgramSim _SPEProgSim;
	 SPURequestsDispatcher m_requestDispatcher;
};

}
}
#endif /*SPEMANAGER_H_*/
