/*
 * Implements send & recv methods of CrsPlatfrmSocket class
 */
#include "../include/M4DCrossPlatformSocket.h"

int16 CrsPlatfrmSocket::instanceCounter = 0;

size_t
CrsPlatfrmSocket::sockRecv( void *buffer, size_t length)
	// throws M4VDSockEx
{
	size_t retval = E_OK;
	#ifdef OS_WIN
	retval = recv( this->sockDescriptor, (char *)buffer, (int) length, 0);
	if ( retval == 0 || retval == WSAECONNRESET ) {
      printf( "Connection Closed.\n");
    }
	#endif
	return retval;
}

size_t
CrsPlatfrmSocket::sockSend( const void *buffer, size_t length)
	//throw (const char *)
{
	size_t retval = E_OK;
	#ifdef OS_WIN
		retval = send( this->sockDescriptor, (const char *)buffer, (int) length, 0);
		if( retval == SOCKET_ERROR)
		{
			throw "Socket error!\n"; 
			int moreSpecErr = WSAGetLastError();
			switch( moreSpecErr)
			{
			case WSAETIMEDOUT:
			case WSAECONNABORTED:
			case WSAEHOSTUNREACH:
			case WSAESHUTDOWN:
				break;
			}
		}
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

CrsPlatfrmSocket::~CrsPlatfrmSocket()
{
#ifdef OS_WIN
	// shutdown socket
	if( shutdown( this->sockDescriptor, SD_SEND) == SOCKET_ERROR)
	{
        printf("shutdown failed: %d\n", WSAGetLastError());
    }

    // cleanup
	closesocket( this->sockDescriptor);

	// decrease instance counter
	instanceCounter--;
	if( instanceCounter == 0)
	{
		// deinit winsock
		WSACleanup();
	}
#endif
}