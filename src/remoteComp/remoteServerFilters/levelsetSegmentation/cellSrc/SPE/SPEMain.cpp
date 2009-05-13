
#include "common/Types.h"
//#include "common/Debug.h"
#include "tools/SPEdebug.h"
#include "tools/DMAGate.h"
//#include "tools/VectorNoExcepts.h"
//#include "vnl/vnl_vector.h"
#include "vnl_math.h"
//#include "itkNumericTraits.h"
#include "updateCalculation/updateCalculatorSPE.h"
#include "applyUpdateCalc/applyUpdateCalculator.h"
//#include "neighborhoodCell.h"
//#include "neighbourhoodIterator.h"

#include <spu_mfcio.h>

//
//typedef float32	TFeatureElement;
//typedef float32 TValueElement;

using namespace M4D::Cell;

int main(unsigned long long speid,
         unsigned long long argp, 
         unsigned long long envp __attribute__ ((unused)))
{
	// config structures
	ConfigStructures _Confs __attribute__ ((aligned (128)));
	RunConfiguration _runConf __attribute__ ((aligned (128)));	
	CalculateChangeAndUpdActiveLayerConf _cChangeConfig __attribute__ ((aligned (128)));
	PropagateValuesConf _propValConfig __attribute__ ((aligned (128)));
	
	// calculator objects
	UpdateCalculatorSPE updateCalculator;
	ApplyUpdateSPE _applyUpdateCalc;
	
	uint32_t mailboxVal;
	uint32 dt;

#ifdef USE_TIMER
  uint64_t start, time_working;
  spu_slih_register (MFC_DECREMENTER_EVENT, spu_clock_slih);
  spu_clock_start();
  start = spu_clock_read();
#endif /* USE_TIMER */

  printf ("This is SPE %lld speaking ... \n", speid);
  
  unsigned int tag = DMAGate::Get((void*)argp, &_Confs, sizeof (ConfigStructures));
  mfc_write_tag_mask (1 << tag);
  mfc_read_tag_status_all ();
    
  tag = DMAGate::Get(_Confs.runConf, &_runConf, sizeof (RunConfiguration));
  mfc_write_tag_mask (1 << tag);
  mfc_read_tag_status_all ();
    
    float32 retval = 0;
  do 
  {
	  mailboxVal = spu_readch(SPU_RdInMbox);
	  switch( (ESPUCommands) mailboxVal)
	  {
	  case CALC_CHANGE:
		  printf ("CALC_CHANGE received\n");
		  // calculate and return retval
		  //retval = updateCalculator.CalculateChange();
		  spu_writech(SPU_WrOutMbox, (uint32_t) JOB_DONE);
		  spu_writech(SPU_WrOutMbox, (uint32_t) retval);
		  break;
	  case CALC_UPDATE:
		  printf ("CALC_UPDATE received\n");
		  dt = spu_readch(SPU_RdInMbox);
		  //retval = _applyUpdateCalc.ApplyUpdate(*((float32 *) &dt));
		  spu_writech(SPU_WrOutMbox, (uint32_t) JOB_DONE);
		  spu_writech(SPU_WrOutMbox, (uint32_t) retval);
		  break;
	  case CALC_PROPAG_VALS:
		  //_applyUpdateCalc.PropagateAllLayerValues();
		  spu_writech(SPU_WrOutMbox, (uint32_t) JOB_DONE);
		  spu_writech(SPU_WrOutMbox, (uint32_t) retval);
		  printf ("CALC_UPDATE received\n");
		  break;
	  case QUIT:
		  printf ("QUIT received\n");
		  spu_writech(SPU_WrOutMbox, (uint32_t) JOB_DONE);
	  	  break;
	  }
  } while(mailboxVal == QUIT);


#ifdef USE_TIMER
  time_working = (spu_clock_read() - start);
  spu_clock_stop();
  printf ("SPE time_working = %lld\n", time_working);
#endif /* USE_TIMER */

  return 0;
}
