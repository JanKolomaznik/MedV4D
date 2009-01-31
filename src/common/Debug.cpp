/**
 *  @ingroup common
 *  @file Debug.cpp
 *  @author Jan Kolomaznik
 */
#include "Debug.h"

#ifdef DEBUG_LEVEL

#include <iostream>
#include <fstream>

//std::ofstream debugFile( "Debug.txt" );
//std::ostream *pdout = &debugFile;
std::ostream *pdout = &(std::cout);

std::string	____EXCEPTION_FILE_NAME = "NOT AVAILABLE";
int		____EXCEPTION_LINE_NUMBER = -1;

#else

std::string	____EXCEPTION_FILE_NAME = "AVAILABLE ONLY IN DEBUG MODE";
int		____EXCEPTION_LINE_NUMBER = -2;

#endif /*DEBUG_LEVEL*/

void
ResetExceptionInfo()
{
	____EXCEPTION_FILE_NAME = "NOT AVAILABLE";
	____EXCEPTION_LINE_NUMBER = -1;
}
