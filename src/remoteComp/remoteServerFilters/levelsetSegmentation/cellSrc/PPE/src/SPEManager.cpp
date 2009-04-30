#include "common/Common.h"
#include "../SPEManager.h"
#include <iostream>


using namespace M4D::Cell;

///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL
extern spe_program_handle_t SPEMain; // handle to SPE program

///////////////////////////////////////////////////////////////////////////////

void *ppu_pthread_function(void *arg) 
{
	unsigned int entry = SPE_DEFAULT_ENTRY;
	
	Tppu_pthread_data *datap = (Tppu_pthread_data *)arg;
	
	std::cout << "Running SPE thread with param=" << datap->argp << std::endl;

	if (spe_context_run(datap->spe_ctx, &entry, 0, datap->argp, NULL, NULL) < 0) {
		perror("Failed running context");
		//exit (1);
	}
	pthread_exit(NULL);
}
#endif
///////////////////////////////////////////////////////////////////////////////

SPEManager::SPEManager() {
	/* Determine the number of SPE threads to create.   */
	speCount = 1;//spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);

#ifdef FOR_CELL
	data = new Tppu_pthread_data[speCount];
#endif
}

///////////////////////////////////////////////////////////////////////////////

void
SPEManager::InitProgramProps(void)
{	
	m_requestDispatcher._workManager = _workManager;
	
	_SPEProgSim.applyUpdateCalc.m_layerGate.dispatcher = &m_requestDispatcher;
	
	// setup apply update
	_SPEProgSim.applyUpdateCalc.commonConf =  &_workManager->GetConfSructs()[0].runConf;
	_SPEProgSim.applyUpdateCalc.m_stepConfig = &_workManager->GetConfSructs()[0].calcChngApplyUpdateConf;	
	_SPEProgSim.applyUpdateCalc.m_propLayerValuesConfig = &_workManager->GetConfSructs()[0].propagateValsConf;	
	
	// and update solver
	_SPEProgSim.updateSolver.m_Conf = &_workManager->GetConfSructs()[0].runConf;
	_SPEProgSim.updateSolver.m_stepConfig = &_workManager->GetConfSructs()[0].calcChngApplyUpdateConf;
	_SPEProgSim.updateSolver.Init();
}

///////////////////////////////////////////////////////////////////////////////

SPEManager::~SPEManager() {
	//TODO stop the SPUs
#ifdef FOR_CELL
	ESPUCommands quitCommand = QUIT;
	SendCommand(quitCommand);
	
	// wait for thread termination
	for (uint32 i=0; i<speCount; i++) {
	    if (pthread_join (data[i].pthread, NULL)) {
	    	D_PRINT ("Failed joining thread");
	    }
	}

	/* Destroy contexts */
	for (uint32 i=0; i<speCount; i++) {
		if (spe_context_destroy(data[i].spe_ctx) != 0) {
			D_PRINT("Failed destroying context");
			//exit (1);
		}
	}	


	delete [] data;
#endif
}

///////////////////////////////////////////////////////////////////////////////
#ifdef FOR_CELL
void
SPEManager::SendCommand(enum ESPUCommands &cmd)
{
	uint32 result;
	for (uint32 i=0; i<speCount; i++) {
		D_PRINT("Write to SPE no: " << i << "'s mailbox, data=" << cmd);
		result = spe_in_mbox_write(data[i].spe_ctx, (uint32*) &cmd, 1, SPE_MBOX_ANY_NONBLOCKING);
		if(result == (uint32) -1)
			; //TODO except
	}
}

///////////////////////////////////////////////////////////////////////////////

void
SPEManager::WaitForCommanResult()
{
	uint32 dataRead;
	for (uint32 i=0; i<speCount; i++) {
		D_PRINT("Read mailbox of " << i << "SPU, waiting...");
		while (spe_out_mbox_status(data[i].spe_ctx) < 1);
		spe_out_mbox_read(data[i].spe_ctx, &dataRead, 1);
		D_PRINT("Read: " << dataRead);
	}
}

///////////////////////////////////////////////////////////////////////////////

void SPEManager::RunSPEs(RunConfiguration *conf) {
	for (uint32 i=0; i<speCount; i++) {
		/* Create SPE context */
		if ((data[i].spe_ctx = spe_context_create(0, NULL)) == NULL) {
			perror("Failed creating context");
			exit(1);
		}
		/* Load SPE program into the SPE context */
		if (spe_program_load(data[i].spe_ctx, &SPEMain)) {
			perror("Failed loading program");
			exit(1);
		}
		/* Initialize context run data */
		data[i].argp = conf;
		/* Create pthread for each of the SPE conexts */
		if (pthread_create(&data[i].pthread, NULL, &ppu_pthread_function, &data[i])) {
			perror("Failed creating thread");
		}
	}
}
#endif
///////////////////////////////////////////////////////////////////////////////
