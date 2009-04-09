#ifndef SPEMANAGER_H_
#define SPEMANAGER_H_

#include <pthread.h>
#include <libspe2.h>

#include "../SPE/configStructures.h"

namespace M4D {
namespace Cell {

struct Tppu_pthread_data
{
  spe_context_ptr_t spe_ctx;
  pthread_t pthread;
  void *argp;
};

class SPEManager
{
public:
	SPEManager();
	~SPEManager();
	
	void RunSPEs(RunConfiguration *conf);
	void SendCommand(ESPUCommands &cmd);
	void WaitForCommanResult();
	
private:
	uint32 speCount;	  
	Tppu_pthread_data *data;
};

}
}
#endif /*SPEMANAGER_H_*/
