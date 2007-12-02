//////////////////////////////////////////////////
// Contains some common declarations for communicaton
// between client & server
//////////////////////////////////////////////////

// port at that server listens
#define SERVER_PORT 44444

// messages identifications
typedef enum eMessageType {
	MESS_DATA_REQUEST,		// client wants some data
	MESS_DATA_SEND,			// sevrver sends requested data
	MESS_DATA_STORE,		// client wants to sotre some data
	MESS_PING,				// ping to see if server is alive
	/* ... */
} MessageType;

// distinguish data requests
typedef enum eRequestType {
	REQTYPE_BITMAP,
	REQTYPE_TENSORMAP,
	/* ... */
} RequestType;