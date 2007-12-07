#ifndef CROSS_PLATFORM_SERVER_SOCK_H
#define CROSS_PLATFORM_SERVER_SOCK_H

#include "../include/M4DCrossPlatformSocket.h"

class CrsPlatfrmServerSocket : CrsPlatfrmSocket
{
public:
	// creates & bind sock to sddress
	CrsPlatfrmServerSocket( void) {}

	~CrsPlatfrmServerSocket( void) {}

	retval_t Create( std::string address, int16 port);
};

#endif
