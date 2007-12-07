/////////////////////////////////////////////////////
// MedV4D project (Data storage)
//
// main.c
/////////////////////////////////////////////////////

#include "server.h"

using namespace dataStorage;

// entry point
int32 main( int argc, char *argv[ ], char *envp[ ] )
{
	Server server;

	server.Start();

	return 0;
}