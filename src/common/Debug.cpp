#include "Debug.h"

#include <iostream>

#ifdef DEBUG_LEVEL

std::ostream *pdout = &(std::cout);

#endif /*DEBUG_LEVEL*/

