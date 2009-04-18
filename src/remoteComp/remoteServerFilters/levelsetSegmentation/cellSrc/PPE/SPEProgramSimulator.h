#ifndef SPEPROGRAMSIMULATOR_H_
#define SPEPROGRAMSIMULATOR_H_

#include "../SPE/updateCalculation/updateCalculatorSPE.h"
#include "../SPE/applyUpdateCalc/applyUpdateCalculator.h"

namespace M4D {
namespace Cell {

class SPUProgramSim
{
public:
	
	typedef M4D::Cell::UpdateCalculatorSPE TUpdateCalculatorSPE;
	TUpdateCalculatorSPE updateSolver;	
	ApplyUpdateSPE applyUpdateCalc;
};

}
}
#endif /*SPEPROGRAMSIMULATOR_H_*/
