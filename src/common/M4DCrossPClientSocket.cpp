
#include "../include/M4DCrossPClientSocket.h"
#include "../include/M4DTransportDefs.h"

void
CrsPlatfrmClientSocket::create( void)
	//throw( const int8*)
{
#ifdef OS_WIN
	this->sockDescriptor = INVALID_SOCKET;
    struct addrinfo *result = NULL,
                    *ptr = NULL,
                    hints;

    int iResult;

	ZeroMemory( &hints, sizeof(hints) );
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

	// convert port number to string
	char strPort[8];
	_ltoa_s<8>( SERVER_PORT, strPort, 10);

    // Resolve the server address and port
    iResult = getaddrinfo( serverAddress.c_str(), strPort, &hints, &result);
    if ( iResult != 0 ) {
        throw "getaddrinfo failed:\n";
    }

    // Attempt to connect to an address until one succeeds
    for(ptr=result; ptr != NULL ;ptr=ptr->ai_next) {

        // Create a SOCKET for connecting to server
		this->sockDescriptor = 
			WSASocket( 
				ptr->ai_family,
				ptr->ai_socktype,
				ptr->ai_protocol,
				NULL, 0,
				WSA_FLAG_OVERLAPPED);
        if (this->sockDescriptor == INVALID_SOCKET) {
			char retStr[64];
            sprintf_s( retStr, 64, "Error at socket(): %ld\n", WSAGetLastError());
            freeaddrinfo( result);
			throw retStr;
        }

        // Connect to server.
        iResult = connect( this->sockDescriptor, ptr->ai_addr, (int)ptr->ai_addrlen);
        if (iResult == SOCKET_ERROR) {
            closesocket( this->sockDescriptor);
            this->sockDescriptor = INVALID_SOCKET;
            continue;
        }
        break;
    }

    freeaddrinfo( result);

    if( this->sockDescriptor == INVALID_SOCKET) {
        throw "Unable to connect to server!\n";
    }

#endif
}

void
CrsPlatfrmClientSocket::create( SOCKET_T s)
	throw ()
{
	this->sockDescriptor = s;
}