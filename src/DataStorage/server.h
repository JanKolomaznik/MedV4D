#ifndef SERVER_H
#define SERVER_H

#include "M4DCrossPServerSock.h"

namespace dataStorage
{

	class Server
	{
		// listening socket
		CrsPlatfrmServerSocket recvSock;

	public:
		Server() {}
		~Server( void) {}

		void Start( void);
		
		// creates new thread that will communicate with client
	private:
		void AcceptConnection( void);
	};

}

#endif