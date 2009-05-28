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

using namespace M4D::Cell;

#define INT_TO_FLOAT(x) (*((float32 *) &x))
#define FLOAT_TO_INT(x) (*((uint32_t *) &x))

int main(unsigned long long speid, unsigned long long argp,
		unsigned long long envp __attribute__ ((unused)))
{
	// config structures
	ConfigStructures _Confs __attribute__ ((aligned (128)));

	SharedResources _sharedRes;

	// calculator objects
	UpdateCalculatorSPE updateCalculator(&_sharedRes);
	ApplyUpdateSPE _applyUpdateCalc(&_sharedRes);

	uint32_t mailboxVal;
	uint32 dt;

#ifdef USE_TIMER
	uint64_t start, time_working;
	spu_slih_register (MFC_DECREMENTER_EVENT, spu_clock_slih);
	spu_clock_start();
	start = spu_clock_read();
#endif /* USE_TIMER */

	printf("This is SPE %lld speaking ... \n", speid);

	unsigned int tag = DMAGate::GetTag();
	DMAGate::Get(argp, &_Confs, sizeof(ConfigStructures), tag);
	mfc_write_tag_mask(1 << tag);
	mfc_read_tag_status_all();

	DMAGate::Get(_Confs.runConf, &_sharedRes._runConf,
			sizeof(RunConfiguration), tag);
	mfc_write_tag_mask(1 << tag);
	mfc_read_tag_status_all();
	DMAGate::ReturnTag(tag);

	updateCalculator.Init();

	// do work loops
	float32 retval = 0;
	do
	{
		mailboxVal = spu_readch(SPU_RdInMbox);
		switch ( (ESPUCommands) mailboxVal)
		{
		case CALC_CHANGE:
			printf("CALC_CHANGE received\n");
			tag = DMAGate::GetTag();
			DMAGate::Get(_Confs.calcChngApplyUpdateConf,
					&_sharedRes._changeConfig,
					sizeof(CalculateChangeAndUpdActiveLayerConf),
					tag);
			mfc_write_tag_mask(1 << tag);
			mfc_read_tag_status_all();
			DMAGate::ReturnTag(tag);

			// calculate and return retval
			retval = updateCalculator.CalculateChange();
			spu_writech(SPU_WrOutMbox, (uint32_t) JOB_DONE);
			spu_writech(SPU_WrOutMbox, FLOAT_TO_INT(retval));
			break;

		case CALC_UPDATE:
			printf("CALC_UPDATE received\n");
			// trasfer step configs
			tag = DMAGate::GetTag();
			DMAGate::Get(_Confs.calcChngApplyUpdateConf,
					&_sharedRes._changeConfig,
					sizeof(CalculateChangeAndUpdActiveLayerConf),
					tag);
			mfc_write_tag_mask(1 << tag);
			mfc_read_tag_status_all();
			DMAGate::Get(_Confs.propagateValsConf,
					&_sharedRes._propValConfig, sizeof(PropagateValuesConf),
					tag);
			mfc_write_tag_mask(1 << tag);
			mfc_read_tag_status_all();
			DMAGate::ReturnTag(tag);

			dt = spu_readch(SPU_RdInMbox);
			retval = _applyUpdateCalc.ApplyUpdate(INT_TO_FLOAT(dt));
			spu_writech(SPU_WrOutMbox, (uint32_t) JOB_DONE);
			spu_writech(SPU_WrOutMbox, FLOAT_TO_INT(retval));
			break;

		case CALC_PROPAG_VALS:
			printf("CALC_PROPAG_VALS received\n");
			DMAGate::GetTag();
			DMAGate::Get(_Confs.propagateValsConf,
					&_sharedRes._propValConfig, sizeof(PropagateValuesConf),
					tag);
			mfc_write_tag_mask(1 << tag);
			mfc_read_tag_status_all();
			DMAGate::ReturnTag(tag);

			_applyUpdateCalc.PropagateAllLayerValues();
			spu_writech(SPU_WrOutMbox, (uint32_t) JOB_DONE);
			// just for something to be writen
			spu_writech(SPU_WrOutMbox, (uint32_t) retval);	
			break;

		case QUIT:
			printf("QUIT received\n");
			spu_writech(SPU_WrOutMbox, (uint32_t) JOB_DONE);
			break;
		}
	} while (mailboxVal != QUIT);

#ifdef USE_TIMER
	time_working = (spu_clock_read() - start);
	spu_clock_stop();
	printf ("SPE time_working = %lld\n", time_working);
#endif /* USE_TIMER */

	printf("SPE %lld quitting ... \n", speid);

	return 0;
}
