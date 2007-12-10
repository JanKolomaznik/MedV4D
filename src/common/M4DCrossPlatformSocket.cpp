/*
 * Implements send & recv methods of CrsPlatfrmSocket class
 */
#include "../include/M4DCrossPlatformSocket.h"

int16 CrsPlatfrmSocket::instanceCounter = 0;

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

CrsPlatfrmSocket::CrsPlatfrmSocket()
{
	#ifdef OS_WIN
	if( instanceCounter == 0)
	{
		// Initialize Winsock
		WSADATA wsaData;		
		int iResult = WSAStartup( MAKEWORD(2,2), &wsaData);
		if (iResult != 0) {
			printf("WSAStartup failed: %d\n", iResult);
		}
	}	
	#endif

	instanceCounter++;
}

CrsPlatfrmSocket::CrsPlatfrmSocket( SOCKET_T s)
	 : sockDescriptor( s)
{
	// call CrsPlatfrmSocket() and how??
}

CrsPlatfrmSocket::~CrsPlatfrmSocket()
{
	instanceCounter--;

	#ifdef OS_WIN
	if( instanceCounter == 0)
	{
		// deinit winsock
		WSACleanup();
	}
	#endif
}