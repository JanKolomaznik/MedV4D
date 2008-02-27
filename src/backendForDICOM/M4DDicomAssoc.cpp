
#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"

#include "M4DDicomAssoc.h"
#include "main.h"
#include "Debug.h"

///////////////////////////////////////////////////////////////////////

// shared address container
M4DDicomAssociation::AddressContainer M4DDicomAssociation::addressContainer;

///////////////////////////////////////////////////////////////////////

M4DDicomAssociation::DICOMAddress *
M4DDicomAssociation::GetAddress( string addressPointer) throw (...)
{
	AddressContainer::iterator it = addressContainer.find( addressPointer);
	if( it == addressContainer.end())
		throw new bad_exception("No such address in container!");
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
M4DDicomAssociation::LoadOneAddress( ifstream &f) throw (...)
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
		if( f.eof() ) throw new bad_exception("Unexpected EOF!");
		addr->callingAPTitle = string( line);

		// read called server name
		FindNonCommentLine( f, line);
		if( f.eof() ) throw new bad_exception("Unexpected EOF!");
		addr->calledServerName = string( line);

		// read called server port
		FindNonCommentLine( f, line);
		if( f.eof() ) throw new bad_exception("Unexpected EOF!");
		sscanf( line.c_str(), "%d ", &addr->calledServerPort);
		if( addr->calledServerPort < 0 ||
			addr->calledServerPort > 65535)
			throw new bad_exception("Port is not in normal range!");

		// read called AT (DB name)
		FindNonCommentLine( f, line);
		if( f.eof() ) throw new bad_exception("Unexpected EOF!");
		addr->calledDBName = string( line);

		// read desired transfer model
		FindNonCommentLine( f, line);
		if( f.eof() ) throw new bad_exception("Unexpected EOF!");
		addr->transferModel = string( line);

		// read desired transfer syntax
		FindNonCommentLine( f, line);
		if( f.eof() ) throw new bad_exception("Unexpected EOF!");
		sscanf( line.c_str(), "%d ", (int *)&addr->transferSyntax);
		if( (int) addr->transferSyntax < 0 ||
			(int) addr->transferSyntax > 3)
			throw new bad_exception("Transfer syntax is incorrect!");

	} catch( bad_exception *) {
		delete addr;
		throw;
	}

	addressContainer.insert( 
		AddressContainer::value_type( addressName, addr) );
}

///////////////////////////////////////////////////////////////////////

void 
M4DDicomAssociation::InitAddressContainer( void) throw (...)
{
	std::ifstream f("config.cfg");
	if( f.fail() )
		throw new bad_exception("config file not found!!");

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

M4DDicomAssociation::M4DDicomAssociation( T_ASC_Network *net, string assocAddrID)
	throw (...)
{
	// check if we have loaded config file
	if( addressContainer.empty() )
		InitAddressContainer();

	// get address from container
	this->m_assocAddr = GetAddress( assocAddrID);

	T_ASC_Parameters *params;
	DIC_NODENAME localHost;
    DIC_NODENAME peerHost;

	OFCondition cond;

	/* initialize asscociation parameters, i.e. create an instance of T_ASC_Parameters*. */
    cond = ASC_createAssociationParameters(&params, ASC_DEFAULTMAXPDU);
    if (cond.bad())	throw new bad_exception("Assoc params could not be created!");

    /* sets this application's title and the called application's title in the params */
    /* structure. The default values to be set here are "STORESCU" and "ANY-SCP". */
	ASC_setAPTitles(params, 
		m_assocAddr->callingAPTitle.c_str(),
		m_assocAddr->calledDBName.c_str(),
		NULL);

    /* Set the transport layer type (type of network connection) in the params */
    /* structure. The default is an insecure connection; where OpenSSL is  */
    /* available the user is able to request an encrypted,secure connection. */
#define SECURED false
    cond = ASC_setTransportLayerType(params, SECURED);
    if (cond.bad()) throw new bad_exception("Security policy setup failed!");

    /* Figure out the presentation addresses and copy the */
    /* corresponding values into the association parameters.*/
    gethostname(localHost, sizeof(localHost) - 1);
	sprintf(peerHost, "%s:%d", 
		m_assocAddr->calledServerName.c_str(), m_assocAddr->calledServerPort);
    ASC_setPresentationAddresses(params, localHost, peerHost);

    /* Set the presentation contexts which will be negotiated */
    /* when the network connection will be established */
	AddPresentationContext( params);

    /* dump presentation contexts if required */
    //if (opt_debug) {
        printf("Request Parameters:\n");
        ASC_dumpParameters(params, COUT);
    //}

    /* create association, i.e. try to establish a network connection to another */
    /* DICOM application. This call creates an instance of T_ASC_Association*. */
    //if (opt_verbose)
        printf("Requesting Association\n");

	cond = ASC_requestAssociation(net, params, &m_assoc);
    if (cond.bad()) {
        if (cond == DUL_ASSOCIATIONREJECTED) {
            T_ASC_RejectParameters rej;
            ASC_getRejectParameters(params, &rej);
            printf("Association Rejected:");
            ASC_printRejectParameters(stderr, &rej);
			throw new bad_exception("Assotiation rejected!");
        } else {
            printf("Association Request Failed:");
            DimseCondition::dump(cond);
			throw new bad_exception("Association Request Failed!");
        }
    }

    /* dump the presentation contexts which have been accepted/refused */
    //if (opt_debug) {
        printf("Association Parameters Negotiated:\n");
        ASC_dumpParameters(params, COUT);
    //}	

    /* count the presentation contexts which have been accepted by the SCP */
    /* If there are none, finish the execution */
    if (ASC_countAcceptedPresentationContexts(params) == 0) {
        throw new bad_exception("No Acceptable Presentation Contexts");
    }

    /* dump general information concerning the establishment of the network connection if required */
    //if (opt_verbose) {
        printf("Association Accepted (Max Send PDV: %lu)\n",
                m_assoc->sendPDVLength);
    //}
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
		throw new bad_exception("ASC_addPresentationContext failed!");
}

///////////////////////////////////////////////////////////////////////

void
M4DDicomAssociation::Abort(void) throw (...)
{
	//if (opt_verbose)
        printf("Aborting Association\n");
	if( ASC_abortAssociation( m_assoc).bad())
	{
        printf("Association Abort Failed:");
		throw new bad_exception("Association Abort Failed");
    }
}

///////////////////////////////////////////////////////////////////////

void
M4DDicomAssociation::Release(void)
{
	//if (opt_verbose)
        printf("Releasing Association\n");
	if( ASC_releaseAssociation(m_assoc).bad())
	{
        printf("Association Release Failed:");
        throw new bad_exception("Association Release Failed");
    }
}

///////////////////////////////////////////////////////////////////////