#ifndef CROSS_PLATFORM_SERVER_SOCK_H
#define CROSS_PLATFORM_SERVER_SOCK_H

#include "../include/M4DCrossPlatformSocket.h"
#include "../include/M4DCrossPClientSocket.h"
#include "../include/M4DTransportDefs.h"

class CrsPlatfrmServerSocket : CrsPlatfrmSocket
{
public:
	// creates & bind sock to sddress
	CrsPlatfrmServerSocket( void) {}

	~CrsPlatfrmServerSocket( void) {}

	void create( void);

	retval_t listenForConn( void);
	void acceptConn( CrsPlatfrmClientSocket *clientSock);
};

#endif
