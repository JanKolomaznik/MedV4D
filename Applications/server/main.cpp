#include "MedV4D/Common/Common.h"
#include <fstream>

#include "remoteComp/serverPart/server.h"

using namespace M4D::Common;
using namespace M4D::RemoteComputing;



int main(int argc, char *argv[]) {
	std::ofstream logFile("Log.txt");
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );

	try {
		asio::io_service asioService;

		Server server(asioService);
		server.AcceptLoop();
		//asioService.run();
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
