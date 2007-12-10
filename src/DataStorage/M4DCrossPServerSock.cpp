
#include <stdlib.h>

#include "./M4DCrossPServerSock.h"

retval_t
CrsPlatfrmServerSocket::create( std::string address, int16 port)
{
#ifdef OS_WIN

	int iRetval;

	struct addrinfo *result = NULL;
					//*ptr = NULL,
	struct addrinfo	hints;

	ZeroMemory( &hints, sizeof(hints) );
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;
	hints.ai_flags = AI_PASSIVE;

	// convert port number to string
	char strPort[8];
	_ltoa_s<8>( port, strPort, 10);

	// Resolve the local address and port to be used by the server
	if( getaddrinfo( NULL, strPort, &hints, &result) != 0 ) {
		printf("getaddrinfo failed: \n");
		return E_FAILED;
	}

	this->sockDescriptor = INVALID_SOCKET;
	this->sockDescriptor = WSASocket(
		result->ai_family,
		result->ai_socktype,
		result->ai_protocol,
		NULL,
		0,
		WSA_FLAG_OVERLAPPED);
	
	if ( this->sockDescriptor == INVALID_SOCKET) {
		printf("WSASocket call failed with error: %ld\n", WSAGetLastError());
		return E_FAILED;
	}

	// bind the socket
	iRetval = bind( this->sockDescriptor, result->ai_addr, (int)result->ai_addrlen);

	freeaddrinfo( result);
	if( iRetval == SOCKET_ERROR) {
        printf("bind failed: %d\n", WSAGetLastError());		
		return E_FAILED;
	}
#endif

	return E_OK;
}

retval_t
CrsPlatfrmServerSocket::listenForConn( void)
{
#ifdef OS_WIN
	if( listen( this->sockDescriptor, SOMAXCONN) == SOCKET_ERROR) {
        printf("listen failed: %d\n", WSAGetLastError());
		return E_FAILED;
	}
	return E_OK;
#endif
}

CrsPlatfrmSocket *
CrsPlatfrmServerSocket::acceptConn( void)
{
	CrsPlatfrmSocket *clientSocket = NULL;
#ifdef OS_WIN
	SOCKET_T clientSock = accept( this->sockDescriptor, NULL, NULL);
    if ( clientSock == INVALID_SOCKET) {
        printf("accept failed: %d\n", WSAGetLastError());
		return NULL;
    }
	// create new clent socket class
	//clientSocket = new CrsPlatfrmClientSocket( clientSock);
#endif
}