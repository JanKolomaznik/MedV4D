#ifndef CROSS_PLATFORM_CLIENT_SOCK_H
#define CROSS_PLATFORM_CLIENT_SOCK_H

#include "../include/M4DCrossPlatformSocket.h"

class CrsPlatfrmClientSocket : CrsPlatfrmSocket
{
public:
	// creates & bind sock to sddress
	CrsPlatfrmClientSocket( void) {}

	~CrsPlatfrmClientSocket( void) {}

	void create( void);
	void create( SOCKET_T s);
private:
	std::string serverAddress;	// TODO
};

#endif