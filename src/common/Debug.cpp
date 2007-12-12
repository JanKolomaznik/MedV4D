#include "Debug.h"

#include <ostream>

#ifdef DEBUGING_LEVEL

std::ostream *pdout = &(std::cout);

#endif /*DEBUGING_LEVEL*/

