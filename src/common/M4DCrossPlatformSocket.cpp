/*
 * Implements send & recv methods of CrsPlatfrmSocket class
 */
#include "../include/M4DCrossPlatformSocket.h"

ssize_t
CrsPlatfrmSocket::recv( void *buffer, size_t length)
{
	ssize_t retval = E_OK;
	#ifdef OS_WIN
		
	#endif
	return retval;
}

ssize_t
CrsPlatfrmSocket::send( void *buffer, size_t length)
{
	ssize_t retval = E_OK;
	#ifdef OS_WIN
		
	#endif
	return retval;
}