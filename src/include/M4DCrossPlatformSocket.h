#ifndef CROSS_PLATFORM_SOCK_H
#define CROSS_PLATFORM_SOCK_H

#include <string>

#include "../include/M4DCommon.h"

/**
 * abstranct class for using sockets. Server & Client
 * socket will derive it. It wraps OS native calls.
 */
class CrsPlatfrmSocket
{
protected:
	CrsPlatfrmSocket( void) {};
	~CrsPlatfrmSocket( void) {};
	
public:
	ssize_t recv( void *buffer, size_t length);
	ssize_t send( void *buffer, size_t length);

	virtual retval_t Create( std::string address, int16 port) = 0;

protected:
	int32 sockDescriptor;
};

#endif