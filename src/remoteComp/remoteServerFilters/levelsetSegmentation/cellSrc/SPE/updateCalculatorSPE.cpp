
#include "common/Types.h"
#include "updateCalculatorSPE.h"

typedef int16	TFeatureElement;
typedef float32 TInnerElement;

typedef itk::Image<TInnerElement, 3> TInputImage;
typedef itk::Image<TInnerElement, 3> TOutputImage;
typedef itk::Image< TFeatureElement, 3 > 		TFautureImage;

typedef itk::UpdateCalculatorSPE<TInputImage, TOutputImage, TOutputImage> TUpdateCalculatorSPE;

int main(unsigned long long speid __attribute__ ((unused)),
         unsigned long long argp __attribute__ ((unused)), 
         unsigned long long envp __attribute__ ((unused)))
{
	
	TUpdateCalculatorSPE updateCalculator;

#ifdef USE_TIMER
  uint64_t start, time_working;
  spu_slih_register (MFC_DECREMENTER_EVENT, spu_clock_slih);
  spu_clock_start();
  start = spu_clock_read();
#endif /* USE_TIMER */



#ifdef USE_TIMER
  time_working = (spu_clock_read() - start);
  spu_clock_stop();
  printf ("SPE time_working = %lld\n", time_working);
#endif /* USE_TIMER */

  return 0;
}