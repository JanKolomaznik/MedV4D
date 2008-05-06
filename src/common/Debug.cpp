#include "Debug.h"

#ifdef DEBUG_LEVEL

#include <iostream>
#include <fstream>

std::ofstream debugFile( "Debug.txt" );
std::ostream *pdout = &debugFile;
//std::ostream *pdout = &(std::cout);

#endif /*DEBUG_LEVEL*/

