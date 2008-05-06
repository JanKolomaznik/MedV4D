#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"
#include "dcmtk/dcmdata/dcdeftag.h"

#include "main.h"

#include "M4DDicomAssoc.h"
#include "M4DDICOMServiceProvider.h"
#include "AbstractService.h"
#include "MoveService.h"

namespace M4D
{

using namespace ErrorHandling;

namespace DicomInternal {

///////////////////////////////////////////////////////////////////////

void
MoveService::GetQuery( 
		DcmDataset **query,
		const string *patientID,
		const string *studyID,
		const string *setID,
		const string *imageID)
{
	if (*query != NULL) delete *query;
    *query = new DcmDataset;

	if( imageID == NULL)
		DU_putStringDOElement(*query, DCM_QueryRetrieveLevel, "SERIES");
	else
		DU_putStringDOElement(*query, DCM_QueryRetrieveLevel, "IMAGE");

	// patient info
	DU_putStringDOElement(*query, DCM_PatientID, 
		(patientID == NULL) ? NULL : patientID->c_str());

	// study info
    DU_putStringDOElement(*query, DCM_StudyInstanceUID, 
		(studyID == NULL) ? NULL : studyID->c_str());

	// serie info
	DU_putStringDOElement(*query, DCM_SeriesInstanceUID, 
		(setID == NULL) ? NULL : setID->c_str());

	// iamge info
	if( imageID != NULL)
	{
		DU_putStringDOElement(*query, DCM_SOPInstanceUID, imageID->c_str());
	}
}

///////////////////////////////////////////////////////////////////////

MoveService::MoveService()
{
	/* initialize network, i.e. create an instance of T_ASC_Network*. */
#define TIME_OUT 30
#define RECIEVE_PORT 51111
	OFCondition cond = ASC_initializeNetwork(
		NET_ACCEPTORREQUESTOR, RECIEVE_PORT, TIME_OUT, &m_net);
	if (cond.bad()) {
		D_PRINT( "Init network failed!");
	}

	// create assotiation
	m_assocToServer = new M4DDicomAssociation( "move");
}

///////////////////////////////////////////////////////////////////////

MoveService::~MoveService()
{
	delete m_assocToServer;
}

///////////////////////////////////////////////////////////////////////

void
MoveService::MoveImage( 
		const string &patientID,
		const string &studyID,
		const string &setID,
		const string &imageID,
		DcmProvider::DicomObj &rs) 
{
	DcmDataset *query = NULL;
	GetQuery( &query, &patientID, &studyID, &setID, &imageID);

	MoveSupport( query, (void *)&rs, SINGLE_IMAGE);
}

///////////////////////////////////////////////////////////////////////

void
MoveService::MoveImageSet(
		const string &patientID,
		const string &studyID,
		const string &serieID,
		DcmProvider::DicomObjSet &result)
{
	DcmDataset *query = NULL;
	GetQuery( &query, &patientID, &studyID, &serieID, NULL);

	MoveSupport( query, (void *)&result, IMAGE_SET);
}

///////////////////////////////////////////////////////////////////////

void
MoveService::MoveSupport( DcmDataset *query,
	void *data, enum eCallType type) 
{
  try {
	  // request assoc to server
	  m_assocToServer->Request(m_net);
  } catch( ExceptionBase &e) {
    LOG( "C-MOVE operation: " << e.what());
    throw;
  }

	T_ASC_PresentationContextID presId;
    T_DIMSE_C_MoveRQ    req;
    T_DIMSE_C_MoveRSP   rsp;
	DIC_US              msgId = m_assocToServer->GetAssociation()->nextMsgID++;
    DcmDataset          *rspIds = NULL;
    //const char          *sopClass;
    DcmDataset          *statusDetail = NULL;

    /* figure out which of the accepted presentation contexts should be used */
	presId = m_assocToServer->FindPresentationCtx();
	
    //MyCallbackInfo      callbackData;
    //callbackData.assoc = assoc;
    //callbackData.presId = presId;

    req.MessageID = msgId;
    strcpy(req.AffectedSOPClassUID, 
		m_assocToServer->GetAssocAddress()->transferModel.c_str() );
    req.Priority = DIMSE_PRIORITY_MEDIUM;
    req.DataSetType = DIMSE_DATASET_PRESENT;

#define MAX_AE_TITLE_LEN 17
	strncpy( req.MoveDestination,
		m_assocToServer->GetAssocAddress()->callingAPTitle.c_str(),
		MAX_AE_TITLE_LEN); // we are the reciever

#define FIND_OPER_TIMEOUT 0

	// do the work
	OFCondition cond;
	switch( type)
	{
	case SINGLE_IMAGE:
		cond = DIMSE_moveUser(
			m_assocToServer->GetAssociation(),
			presId, &req, query,
			MoveCallback, NULL, 
			DIMSE_BLOCKING, FIND_OPER_TIMEOUT,
			m_net, SubAssocCallback, data,
			&rsp, &statusDetail, &rspIds, false);
		break;

	case IMAGE_SET:
		cond = DIMSE_moveUser(
			m_assocToServer->GetAssociation(),
			presId, &req, query,
			MoveCallback, NULL, 
			DIMSE_BLOCKING, FIND_OPER_TIMEOUT,
			m_net, SubAssocCallbackWholeSet, data,
			&rsp, &statusDetail, &rspIds, false);
		break;
	}    

  if (cond == EC_Normal) 
  {
	  LOG( "Move operation accepted ... image transmission pending ...");
#ifdef _DEBUG
    if (rspIds != NULL) 
    {
        D_PRINT("Response Identifiers:\n");
        rspIds->print(LOUT);
    }
#endif
  } else {
    LOG("Move Failed ...");
  }
  if (statusDetail != NULL) 
  {
    LOG("  Status Detail:\n");
    statusDetail->print(LOUT);
    delete statusDetail;
  }

  if (rspIds != NULL) delete rspIds;

	m_assocToServer->Release();
}

///////////////////////////////////////////////////////////////////////

void
MoveService::AcceptSubAssoc(T_ASC_Network * aNet, T_ASC_Association ** assoc)
	
{
	// this is hardcoded ! Firstly -> no compression or some JPEGs
	E_TransferSyntax prefferedTransferSyntax = EXS_LittleEndianImplicit;

    const char* knownAbstractSyntaxes[] = {
        UID_VerificationSOPClass
    };
    const char* transferSyntaxes[] = { NULL, NULL, NULL, NULL };
    int numTransferSyntaxes;

    OFCondition cond = ASC_receiveAssociation(aNet, assoc, (int)m_maxPDU);
    if (cond.good())
    {
      switch (prefferedTransferSyntax)
      {
        case EXS_LittleEndianImplicit:
          /* we only support Little Endian Implicit */
          transferSyntaxes[0]  = UID_LittleEndianImplicitTransferSyntax;
          numTransferSyntaxes = 1;
          break;
        case EXS_LittleEndianExplicit:
          /* we prefer Little Endian Explicit */
          transferSyntaxes[0] = UID_LittleEndianExplicitTransferSyntax;
          transferSyntaxes[1] = UID_BigEndianExplicitTransferSyntax;
          transferSyntaxes[2]  = UID_LittleEndianImplicitTransferSyntax;
          numTransferSyntaxes = 3;
          break;
        case EXS_BigEndianExplicit:
          /* we prefer Big Endian Explicit */
          transferSyntaxes[0] = UID_BigEndianExplicitTransferSyntax;
          transferSyntaxes[1] = UID_LittleEndianExplicitTransferSyntax;
          transferSyntaxes[2]  = UID_LittleEndianImplicitTransferSyntax;
          numTransferSyntaxes = 3;
          break;
        case EXS_JPEGProcess14SV1TransferSyntax:
          /* we prefer JPEGLossless:Hierarchical-1stOrderPrediction (default lossless) */
          transferSyntaxes[0] = UID_JPEGProcess14SV1TransferSyntax;
          transferSyntaxes[1] = UID_LittleEndianExplicitTransferSyntax;
          transferSyntaxes[2] = UID_BigEndianExplicitTransferSyntax;
          transferSyntaxes[3] = UID_LittleEndianImplicitTransferSyntax;
          numTransferSyntaxes = 4;
          break;
        case EXS_JPEGProcess1TransferSyntax:
          /* we prefer JPEGBaseline (default lossy for 8 bit images) */
          transferSyntaxes[0] = UID_JPEGProcess1TransferSyntax;
          transferSyntaxes[1] = UID_LittleEndianExplicitTransferSyntax;
          transferSyntaxes[2] = UID_BigEndianExplicitTransferSyntax;
          transferSyntaxes[3] = UID_LittleEndianImplicitTransferSyntax;
          numTransferSyntaxes = 4;
          break;
        case EXS_JPEGProcess2_4TransferSyntax:
          /* we prefer JPEGExtended (default lossy for 12 bit images) */
          transferSyntaxes[0] = UID_JPEGProcess2_4TransferSyntax;
          transferSyntaxes[1] = UID_LittleEndianExplicitTransferSyntax;
          transferSyntaxes[2] = UID_BigEndianExplicitTransferSyntax;
          transferSyntaxes[3] = UID_LittleEndianImplicitTransferSyntax;
          numTransferSyntaxes = 4;
          break;
        case EXS_JPEG2000:
          /* we prefer JPEG2000 Lossy */
          transferSyntaxes[0] = UID_JPEG2000TransferSyntax;
          transferSyntaxes[1] = UID_LittleEndianExplicitTransferSyntax;
          transferSyntaxes[2] = UID_BigEndianExplicitTransferSyntax;
          transferSyntaxes[3] = UID_LittleEndianImplicitTransferSyntax;
          numTransferSyntaxes = 4;
          break;
        case EXS_JPEG2000LosslessOnly:
          /* we prefer JPEG2000 Lossless */
          transferSyntaxes[0] = UID_JPEG2000LosslessOnlyTransferSyntax;
          transferSyntaxes[1] = UID_LittleEndianExplicitTransferSyntax;
          transferSyntaxes[2] = UID_BigEndianExplicitTransferSyntax;
          transferSyntaxes[3] = UID_LittleEndianImplicitTransferSyntax;
          numTransferSyntaxes = 4;
          break;
        case EXS_RLELossless:
          /* we prefer RLE Lossless */
          transferSyntaxes[0] = UID_RLELosslessTransferSyntax;
          transferSyntaxes[1] = UID_LittleEndianExplicitTransferSyntax;
          transferSyntaxes[2] = UID_BigEndianExplicitTransferSyntax;
          transferSyntaxes[3] = UID_LittleEndianImplicitTransferSyntax;
          numTransferSyntaxes = 4;
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

        /* accept the Verification SOP Class if presented */
        cond = ASC_acceptContextsWithPreferredTransferSyntaxes(
            (*assoc)->params,
            knownAbstractSyntaxes, DIM_OF(knownAbstractSyntaxes),
            transferSyntaxes, numTransferSyntaxes);

        if (cond.good())
        {
            /* the array of Storage SOP Class UIDs comes from dcuid.h */
            cond = ASC_acceptContextsWithPreferredTransferSyntaxes(
                (*assoc)->params,
                dcmAllStorageSOPClassUIDs, numberOfAllDcmStorageSOPClassUIDs,
                transferSyntaxes, numTransferSyntaxes);
        }
    }
    if (cond.good()) cond = ASC_acknowledgeAssociation(*assoc);
    if (cond.bad()) {
		D_PRINT( "No acceptable image trasfer sytaxes!");
		throw ExceptionBase();
    }
}

///////////////////////////////////////////////////////////////////////

/**
 *	Call back called when image arrive
 */
void
MoveService::StoreSCPCallback(
    /* in */
    void *callbackData,
    T_DIMSE_StoreProgress *progress,    /* progress state */
    T_DIMSE_C_StoreRQ * /*req*/,             /* original store request */
    char * /*imageFileName*/, 
	DcmDataset ** /*imageDataSet*/, /* being received into */
    /* out */
    T_DIMSE_C_StoreRSP * /*rsp*/,            /* final store response */
    DcmDataset ** /*statusDetail*/)
{

    /*if ((opt_abortDuringStore && progress->state != DIMSE_StoreBegin) ||
        (opt_abortAfterStore && progress->state == DIMSE_StoreEnd)) {
        if (opt_verbose) {
            printf("ABORT initiated (due to command line options)\n");
        }
        ASC_abortAssociation(((StoreCallbackData*) callbackData)->assoc);
        rsp->DimseStatus = STATUS_STORE_Refused_OutOfResources;
        return;
    }*/

    if (progress->state == DIMSE_StoreEnd)
    {
		// image recieved !!! Huraaaaa !!!!
		LOG("Image recieved");

		// set loaded flag
		DcmProvider::DicomObj *result = 
			static_cast<DcmProvider::DicomObj *>(callbackData);
		result->Init();
	}
}

///////////////////////////////////////////////////////////////////////

void
MoveService::SubTransferOperationSCP(
	T_ASC_Association **subAssoc, void *data, eCallType type) 
{
    T_DIMSE_Message     msg;
    T_ASC_PresentationContextID presID;

    if (!ASC_dataWaiting(*subAssoc, 0)) /* just in case */
	{
        throw ExceptionBase("No data waiting!");
	}

#define MOVE_OPER_TIMEOUT 30

    OFCondition cond = DIMSE_receiveCommand(
		*subAssoc, DIMSE_NONBLOCKING, MOVE_OPER_TIMEOUT, &presID,
        &msg, NULL);

	T_DIMSE_C_StoreRQ *req;
	DcmProvider::DicomObj *result = NULL;

	DcmProvider::DicomObjSet *set;
	DcmProvider::DicomObj buddy;

    if (cond == EC_Normal) 
	{
        switch (msg.CommandField) {
        case DIMSE_C_STORE_RQ:
			cond = EC_Normal;
			req = &msg.msg.CStoreRQ;	

			switch( type)
			{
			case IMAGE_SET:
				// insert new DICOMObj into container
				set = static_cast<
					DcmProvider::DicomObjSet *>(data);

				// SINCHRONIZE !!! ???
				set->push_back( buddy);
				result = &set->back();
				// SINCHRONIZE !!! ???
				break;

			case SINGLE_IMAGE:
				result = 
					static_cast<DcmProvider::DicomObj *>(data);
				break;
			}

#define WRITE_METAHEADER false

			cond = DIMSE_storeProvider(
				*subAssoc, presID, req, 
				(char *)NULL, WRITE_METAHEADER,
				(DcmDataset **)&result->m_dataset, 
				StoreSCPCallback, (void *)result,
				DIMSE_NONBLOCKING, MOVE_OPER_TIMEOUT);
            break;

        default:
			D_PRINT("Unknown command recieved on sub assotation!");
			throw ExceptionBase();
            break;
        }
    }

	/* clean up on association termination */
	if (cond == DUL_PEERREQUESTEDRELEASE)
	{
		if( ASC_acknowledgeRelease(*subAssoc).bad())
		{
			D_PRINT("Release ACK not sent!");
			throw ExceptionBase();
		}
		ASC_dropSCPAssociation(*subAssoc);
		ASC_destroyAssociation(subAssoc);
	}
	else if (cond == DUL_PEERABORTEDASSOCIATION)
	{
	}
	else if (cond != EC_Normal)
	{
		LOG("DIMSE Failure (aborting sub-association)");
		/* some kind of error so abort the association */
		cond = ASC_abortAssociation(*subAssoc);
	}

	if (cond != EC_Normal)
	{
		ASC_dropAssociation(*subAssoc);
		ASC_destroyAssociation(subAssoc);
	}
}

///////////////////////////////////////////////////////////////////////

/**
 *	Called when STORE-SCU on server side wants to establish transfer
 *	assotiation with us
 */
void
MoveService::SubAssocCallback(void *subOpCallbackData,
        T_ASC_Network *aNet, T_ASC_Association **subAssoc)
{
	SubAssocCallbackSupp( subOpCallbackData, aNet,
		subAssoc, SINGLE_IMAGE);
}

///////////////////////////////////////////////////////////////////////

/**
 *	Called when STORE-SCU on server side wants to establish transfer
 *	assotiation with us
 */
void
MoveService::SubAssocCallbackWholeSet(void *subOpCallbackData,
        T_ASC_Network *aNet, T_ASC_Association **subAssoc)
{
    SubAssocCallbackSupp( subOpCallbackData, aNet,
		subAssoc, IMAGE_SET);
}

///////////////////////////////////////////////////////////////////////

/**
 *	Called when STORE-SCU on server side wants to establish transfer
 *	assotiation with us
 */
void
MoveService::SubAssocCallbackSupp(void *subOpCallbackData,
        T_ASC_Network *aNet, T_ASC_Association **subAssoc, eCallType type)
{
    if (aNet == NULL) return;   /* help no net ! */

    if (*subAssoc == NULL) {
        /* negotiate association */
        AcceptSubAssoc(aNet, subAssoc);
    } else {
        /* be a service class provider */
        SubTransferOperationSCP( subAssoc, subOpCallbackData, type);
    }
}

///////////////////////////////////////////////////////////////////////

/**
 *	Called when QUERY/RETRIEVE SCP send response that another image was sent
 */
void
MoveService::MoveCallback( void * /*callbackData*/,
	T_DIMSE_C_MoveRQ * /*request*/,
    int responseCount, T_DIMSE_C_MoveRSP * /*response*/)
{
	// there is nothing much to do.
	// just logg it
	LOG("Move Response, #" << responseCount);
}

///////////////////////////////////////////////////////////////////////

} // namespace
}