#ifndef CROSS_PLATFORM_SOCK_H
#define CROSS_PLATFORM_SOCK_H

#include <string>

#include "../include/M4DCommon.h"

#ifdef OS_WIN
	#include <winsock2.h>
	#include <ws2tcpip.h>
#endif

// socket descriptor type
#ifdef OS_WIN
	#define SOCKET_T SOCKET
#endif

/**
 * abstranct class for using sockets. Server & Client
 * socket will derive it. It wraps OS native calls.
 */
class CrsPlatfrmSocket
{
protected:
	CrsPlatfrmSocket( void);
	CrsPlatfrmSocket( SOCKET_T s);
	~CrsPlatfrmSocket( void);
	
public:
	ssize_t recv( void *buffer, size_t length);
	ssize_t send( void *buffer, size_t length);

	virtual retval_t create( std::string address, int16 port) = 0;

protected:	
	SOCKET_T sockDescriptor;

	static int16 instanceCounter;
};

#endif