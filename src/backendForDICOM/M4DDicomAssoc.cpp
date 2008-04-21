
#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"

#include "main.h"

#include "M4DDicomAssoc.h"

namespace M4D
{

using namespace ErrorHandling;
///////////////////////////////////////////////////////////////////////

// shared address container
M4DDicomAssociation::AddressContainer M4DDicomAssociation::addressContainer;

///////////////////////////////////////////////////////////////////////

M4DDicomAssociation::DICOMAddress *
M4DDicomAssociation::GetAddress( string addressPointer) 
{
	AddressContainer::iterator it = addressContainer.find( addressPointer);
	if( it == addressContainer.end())
		throw new ExceptionBase("No such address in container!");
	else
		return it->second;
}

/////////////////////////////////////////////////////////////////////

#define MAX_LINE_LEN 128
void
M4DDicomAssociation::FindNonCommentLine( ifstream &f, string &line)
{
	char tmp[MAX_LINE_LEN];

	f >> line;

	while( !f.eof() && line.find("//") == 0)
	{
		// throw away the rest of line since it is comment
		f.getline( tmp, MAX_LINE_LEN);
		f >> line;
	}
}

///////////////////////////////////////////////////////////////////////

void
M4DDicomAssociation::LoadOneAddress( ifstream &f) 
{
	string line;

	FindNonCommentLine( f, line);
	if( f.eof() )
		return;

	string addressName;
	DICOMAddress *addr = new DICOMAddress;

	// read address name (the key to map)
	addressName = line;	

	try {
		
		// read our AP Title
		FindNonCommentLine( f, line);
		if( f.eof() ) throw new ExceptionBase("Unexpected EOF!");
		addr->callingAPTitle = string( line);

		// read called server name
		FindNonCommentLine( f, line);
		if( f.eof() ) throw new ExceptionBase("Unexpected EOF!");
		addr->calledServerName = string( line);

		// read called server port
		FindNonCommentLine( f, line);
		if( f.eof() ) throw new ExceptionBase("Unexpected EOF!");
		sscanf( line.c_str(), "%d ", &addr->calledServerPort);
		if( addr->calledServerPort < 0 ||
			addr->calledServerPort > 65535)
			throw new ExceptionBase("Port is not in normal range!");

		// read called AT (DB name)
		FindNonCommentLine( f, line);
		if( f.eof() ) throw new ExceptionBase("Unexpected EOF!");
		addr->calledDBName = string( line);

		// read desired transfer model
		FindNonCommentLine( f, line);
		if( f.eof() ) throw new ExceptionBase("Unexpected EOF!");
		addr->transferModel = string( line);

		// read desired transfer syntax
		FindNonCommentLine( f, line);
		if( f.eof() ) throw new ExceptionBase("Unexpected EOF!");
		sscanf( line.c_str(), "%d ", (int *)&addr->transferSyntax);
		if( (int) addr->transferSyntax < 0 ||
			(int) addr->transferSyntax > 3)
			throw new ExceptionBase("Transfer syntax is incorrect!");

	} catch( ExceptionBase *) {
		delete addr;
		throw;
	}

	addressContainer.insert( 
		AddressContainer::value_type( addressName, addr) );
}

///////////////////////////////////////////////////////////////////////

void 
M4DDicomAssociation::InitAddressContainer( void) 
{
	std::ifstream f("config.cfg");
	if( f.fail() )
		throw new ExceptionBase("config file not found!!");

	while( !f.eof())
	{
		LoadOneAddress( f);
	}	
}

///////////////////////////////////////////////////////////////////////

M4DDicomAssociation::~M4DDicomAssociation( void)
{
	/* destroy the association, i.e. free memory of T_ASC_Association* structure. This */
    /* call is the counterpart of ASC_requestAssociation(...) which was called above. */
    if( ASC_destroyAssociation( &m_assoc).bad() )
    {
        // TODO hlaska
    }
}

///////////////////////////////////////////////////////////////////////

M4DDicomAssociation::M4DDicomAssociation( string assocAddrID)
	
{
	// check if we have loaded config file
	if( addressContainer.empty() )
		InitAddressContainer();

	// get address from container
	this->m_assocAddr = GetAddress( assocAddrID);

	DIC_NODENAME localHost;
    DIC_NODENAME peerHost;

	/* initialize asscociation parameters, i.e. create an instance of T_ASC_Parameters*. */
    OFCondition cond = ASC_createAssociationParameters(&m_assocParams, ASC_DEFAULTMAXPDU);
    if (cond.bad())	
	{
		D_PRINT( "Assoc params could not be created!");
		throw new ExceptionBase();
	}

    /* sets this application's title and the called application's title in the params */
    /* structure. The default values to be set here are "STORESCU" and "ANY-SCP". */
	ASC_setAPTitles(m_assocParams, 
		m_assocAddr->callingAPTitle.c_str(),
		m_assocAddr->calledDBName.c_str(),
		NULL);

    /* Set the transport layer type (type of network connection) in the params */
    /* structure. The default is an insecure connection; where OpenSSL is  */
    /* available the user is able to request an encrypted,secure connection. */
#define SECURED false
    cond = ASC_setTransportLayerType(m_assocParams, SECURED);
    if (cond.bad())
	{
		D_PRINT( "Security policy setup failed!");
		throw new ExceptionBase();
	}

    /* Figure out the presentation addresses and copy the */
    /* corresponding values into the association parameters.*/
    gethostname(localHost, sizeof(localHost) - 1);
	sprintf(peerHost, "%s:%d", 
		m_assocAddr->calledServerName.c_str(), m_assocAddr->calledServerPort);
    ASC_setPresentationAddresses(m_assocParams, localHost, peerHost);

    /* Set the presentation contexts which will be negotiated */
    /* when the network connection will be established */
	AddPresentationContext( m_assocParams);
}

void
M4DDicomAssociation::Request( T_ASC_Network *net) 
{
	/* dump presentation contexts if required */
    D_PRINT("Request Parameters");
    ASC_dumpParameters(m_assocParams, DOUT);

    /* create association, i.e. try to establish a network connection to another */
    /* DICOM application. This call creates an instance of T_ASC_Association*. */
	D_PRINT("Requesting Association");

	OFCondition cond = ASC_requestAssociation(net, m_assocParams, &m_assoc);
    if (cond.bad()) {
        if (cond == DUL_ASSOCIATIONREJECTED) {
            T_ASC_RejectParameters rej;
            ASC_getRejectParameters(m_assocParams, &rej);
			D_PRINT("Association Rejected due this params:");
            ASC_printRejectParameters(DOUT, &rej);
			throw new ExceptionBase("Assotiation rejected!");
        } else {
            D_PRINT("Association Request Failed");
			throw new ExceptionBase("Association Request Failed!");
        }
    }

    D_PRINT("Association Parameters Negotiated");

    /* count the presentation contexts which have been accepted by the SCP */
    /* If there are none, finish the execution */
    if (ASC_countAcceptedPresentationContexts(m_assocParams) == 0) {
		D_PRINT("No Acceptable Presentation Contexts");
        throw new ExceptionBase("No Acceptable Presentation Contexts");
    }

    /* dump general information concerning the establishment 
	of the network connection if required */
	LOG("Association Accepted (Max Send PDV: " << m_assoc->sendPDVLength << ")");
}

///////////////////////////////////////////////////////////////////////

void
M4DDicomAssociation::AddPresentationContext(T_ASC_Parameters *params)
{
    /*
    ** We prefer to use Explicitly encoded transfer syntaxes.
    ** If we are running on a Little Endian machine we prefer
    ** LittleEndianExplicitTransferSyntax to BigEndianTransferSyntax.
    ** Some SCP implementations will just select the first transfer
    ** syntax they support (this is not part of the standard) so
    ** organise the proposed transfer syntaxes to take advantage
    ** of such behaviour.
    **
    ** The presentation contexts proposed here are only used for
    ** C-FIND and C-MOVE, so there is no need to support compressed
    ** transmission.
    */

    const char* transferSyntaxes[] = { NULL, NULL, NULL };
    int numTransferSyntaxes = 0;

	switch (m_assocAddr->transferSyntax) {
    case EXS_LittleEndianImplicit:
        /* we only support Little Endian Implicit */
        transferSyntaxes[0]  = UID_LittleEndianImplicitTransferSyntax;
        numTransferSyntaxes = 1;
        break;
    case EXS_LittleEndianExplicit:
        /* we prefer Little Endian Explicit */
        transferSyntaxes[0] = UID_LittleEndianExplicitTransferSyntax;
        transferSyntaxes[1] = UID_BigEndianExplicitTransferSyntax;
        transferSyntaxes[2] = UID_LittleEndianImplicitTransferSyntax;
        numTransferSyntaxes = 3;
        break;
    case EXS_BigEndianExplicit:
        /* we prefer Big Endian Explicit */
        transferSyntaxes[0] = UID_BigEndianExplicitTransferSyntax;
        transferSyntaxes[1] = UID_LittleEndianExplicitTransferSyntax;
        transferSyntaxes[2] = UID_LittleEndianImplicitTransferSyntax;
        numTransferSyntaxes = 3;
        break;
    default:
        /* We prefer explicit transfer syntaxes.
         * If we are running on a Little Endian machine we prefer
         * LittleEndianExplicitTransferSyntax to BigEndianTransferSyntax.
         */
        if (gLocalByteOrder == EBO_LittleEndian)  /* defined in dcxfer.h */
        {
            transferSyntaxes[0] = UID_LittleEndianExplicitTransferSyntax;
            transferSyntaxes[1] = UID_BigEndianExplicitTransferSyntax;
        } else {
            transferSyntaxes[0] = UID_BigEndianExplicitTransferSyntax;
            transferSyntaxes[1] = UID_LittleEndianExplicitTransferSyntax;
        }
        transferSyntaxes[2] = UID_LittleEndianImplicitTransferSyntax;
        numTransferSyntaxes = 3;
        break;
    }

    if( ASC_addPresentationContext(
		params, 1, m_assocAddr->transferModel.c_str(),
		transferSyntaxes, numTransferSyntaxes).bad() )
	{
		D_PRINT("ASC_addPresentationContext failed!");
		throw new ExceptionBase("ASC_addPresentationContext failed!");
	}
}

///////////////////////////////////////////////////////////////////////

void
M4DDicomAssociation::Abort(void) 
{
	LOG("Aborting Association");

	if( ASC_abortAssociation( m_assoc).bad())
	{
        D_PRINT("Association Abort Failed");
		throw new ExceptionBase("Association Abort Failed");
    }
}

///////////////////////////////////////////////////////////////////////

void
M4DDicomAssociation::Release(void)
{
    LOG("Releasing Association");

	if( ASC_releaseAssociation(m_assoc).bad())
	{
        D_PRINT("Association Release Failed!");
        throw new ExceptionBase("Association Release Failed");
    }
}

///////////////////////////////////////////////////////////////////////

T_ASC_PresentationContextID
M4DDicomAssociation::FindPresentationCtx( void) throw(...)
{
	T_ASC_PresentationContextID presId = 
		ASC_findAcceptedPresentationContextID(
			m_assoc,
			m_assocAddr->transferModel.c_str() );
    if (presId == 0) {
        D_PRINT( "No presentation context");
        throw new ExceptionBase("No presentation ctx");
    }
	return presId;
}

///////////////////////////////////////////////////////////////////////
}