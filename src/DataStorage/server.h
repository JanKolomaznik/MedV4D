#ifndef SERVER_H
#define SERVER_H

#include "M4DCommon.h"
#include "M4DTransportDefs.h"

namespace dataStorage {

	class Server
	{
		// listening socket
		M4D_SOCKET recvSock;

		Server();

		public void Start();
		
		// creates new thread that will communicate with client
		private void AcceptConnection();
	}

}

#endif