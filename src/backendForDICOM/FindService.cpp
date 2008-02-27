#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"
#include "dcmtk/dcmdata/dcdeftag.h"

#include "M4DDicomAssoc.h"
#include "M4DDICOMServiceProvider.h"
#include "FindService.h"


///////////////////////////////////////////////////////////////////////

void
M4DFindService::GetSeriesQuery( DcmDataset **query)
{
	if (*query != NULL) delete *query;
    *query = new DcmDataset;

    DU_putStringDOElement(*query, DCM_QueryRetrieveLevel, "SERIES");
	// patient info
	DU_putStringDOElement(*query, DCM_PatientID, NULL);
	DU_putStringDOElement(*query, DCM_PatientsName, NULL);
	DU_putStringDOElement(*query, DCM_PatientsSex, NULL);
	DU_putStringDOElement(*query, DCM_PatientsBirthDate, NULL);

	// study info
    DU_putStringDOElement(*query, DCM_StudyInstanceUID, NULL);

	// serie info
    DU_putStringDOElement(*query, DCM_SeriesInstanceUID, NULL);
    DU_putStringDOElement(*query, DCM_Modality, NULL);
    DU_putStringDOElement(*query, DCM_SeriesNumber, NULL);
}

///////////////////////////////////////////////////////////////////////

void
M4DFindService::GetImageQuery(
	DcmDataset **query, std::string studyID, std::string serieID)
{
}

///////////////////////////////////////////////////////////////////////

M4DFindService::M4DFindService()
{
	/*
    ** Don't let dcmdata remove tailing blank padding or perform other
    ** maipulations.  We want to see the real data.
    */
    dcmEnableAutomaticInputDataCorrection.set(OFFalse);

	/* make sure data dictionary is loaded */
    /*if (!dcmDataDict.isDictionaryLoaded()) {
        fprintf(stderr, "Warning: no data dictionary loaded, check environment variable: %s\n",
                DCM_DICT_ENVIRONMENT_VARIABLE);
    }*/

#ifdef HAVE_WINSOCK_H
    WSAData winSockData;
    /* we need at least version 1.1 */
    WORD winSockVersionNeeded = MAKEWORD( 1, 1 );
    WSAStartup(winSockVersionNeeded, &winSockData);
#endif

	/* initialize network, i.e. create an instance of T_ASC_Network*. */
#define TIME_OUT 30
    OFCondition cond = ASC_initializeNetwork(NET_REQUESTOR, 0, TIME_OUT, &net);
    if (cond.bad()) {
        DimseCondition::dump(cond);
        exit(1);
    }

	// create assotiation
	m_assocToServer = new M4DDicomAssociation( net, "find");
}

///////////////////////////////////////////////////////////////////////

M4DFindService::~M4DFindService()
{
	delete m_assocToServer;

	/* drop the network, i.e. free memory of T_ASC_Network* structure. This call */
    /* is the counterpart of ASC_initializeNetwork(...) which was called above. */
	if( ASC_dropNetwork(&net).bad() )
    {
        // TODO hlaska
    }

#ifdef HAVE_WINSOCK_H
    WSACleanup();
#endif

}

///////////////////////////////////////////////////////////////////////

void
M4DFindService::Find( M4DDicomServiceProvider::ResultSet &rs) throw (...)
{
	// do the ral work	////////////////////////////////////////
	DIC_US msgId = m_assocToServer->GetAssociation()->nextMsgID++;
    T_ASC_PresentationContextID presId;
    T_DIMSE_C_FindRQ req;
    T_DIMSE_C_FindRSP rsp;
    DcmDataset *statusDetail = NULL;
    //MyCallbackInfo callbackData;

    /* figure out which of the accepted presentation contexts should be used */
    presId = ASC_findAcceptedPresentationContextID(
		m_assocToServer->GetAssociation(),
		m_assocToServer->GetAssocAddress()->transferModel.c_str() );
    if (presId == 0) {
        //errmsg("No presentation context");
        throw new bad_exception("No presentation context");
    }

    /* prepare the transmission of data */
    bzero((char*)&req, sizeof(req));
    req.MessageID = msgId;
	strcpy(req.AffectedSOPClassUID, 
		m_assocToServer->GetAssocAddress()->transferModel.c_str() );
    req.DataSetType = DIMSE_DATASET_PRESENT;
    req.Priority = DIMSE_PRIORITY_LOW;

    /* prepare the callback data */
    //callbackData.assoc = assoc;
    //callbackData.presId = presId;

	// create data set representing the query
	DcmDataset *dataSet = NULL;
	this->GetSeriesQuery( &dataSet);

    /* if required, dump some more general information */
    //if (opt_verbose) {
        printf("Find SCU RQ: MsgID %d\n", msgId);
        printf("REQUEST:\n");
        dataSet->print(COUT);
        printf("--------\n");
    //}

#define FIND_OPER_TIMEOUT 0

    /* finally conduct transmission of data */
	OFCondition cond = DIMSE_findUser(
		m_assocToServer->GetAssociation(),
		presId, &req, dataSet,
		&M4DFindService::ProgressCallback, (void *)&rs,
		DIMSE_BLOCKING, FIND_OPER_TIMEOUT,
		&rsp, &statusDetail);


    /* dump some more general information */
    if (cond == EC_Normal) {
        /*if (opt_verbose) {
            DIMSE_printCFindRSP(stdout, &rsp);
        } else {*/
            if (rsp.DimseStatus != STATUS_Success) {
                printf("Response: %s\n", DU_cfindStatusString(rsp.DimseStatus));
            }
        //}
    } else {
        printf("Find Failed, query keys:");
        dataSet->print(COUT);
        DimseCondition::dump(cond);
    }

    /* dump status detail information if there is some */
    if (statusDetail != NULL) {
        printf("  Status Detail:\n");
        statusDetail->print(COUT);
        delete statusDetail;
    }

}

///////////////////////////////////////////////////////////////////////

void
M4DFindService::ProgressCallback(
        void *callbackData,
        T_DIMSE_C_FindRQ *request,
        int responseCount,
        T_DIMSE_C_FindRSP *rsp,
        DcmDataset *responseIdentifiers
        )
    /*
     * This function.is used to indicate progress when findscu receives search results over the
     * network. This function will simply cause some information to be dumped to stdout.
     *
     * Parameters:
     *   callbackData        - [in] data for this callback function
     *   request             - [in] The original find request message.
     *   responseCount       - [in] Specifies how many C-FIND-RSP were received including the current one.
     *   rsp                 - [in] the C-FIND-RSP message which was received shortly before the call to
     *                              this function.
     *   responseIdentifiers - [in] Contains the record which was received. This record matches the search
     *                              mask of the C-FIND-RQ which was sent.
     */
{

	DcmElement *e = new DcmByteString( DCM_PatientsName);
	OFString str;

	//////////////////////////////////////////////////
	// Parse the response
	//////////////////////////////////////////////////
	M4DDicomServiceProvider::M4DTableRow *row = new M4DDicomServiceProvider::M4DTableRow;

	responseIdentifiers->findAndGetOFString( DCM_PatientsName, str);
	row->patientName = str.c_str();

	responseIdentifiers->findAndGetOFString( DCM_PatientID, str);
	row->patentID = str.c_str();

	responseIdentifiers->findAndGetOFString( DCM_PatientsBirthDate, str);
	row->patientBirthDate = str.c_str();

	responseIdentifiers->findAndGetOFString( DCM_PatientsSex, str);
	row->patientSex = (str == "M");	// M = true

	// study info
	responseIdentifiers->findAndGetOFString( DCM_StudyID, str);
	row->studyID = str.c_str();

	responseIdentifiers->findAndGetOFString( DCM_StudyDate, str);
	row->studyDate = str.c_str();

	responseIdentifiers->findAndGetOFString( DCM_Modality, str);
	row->modality = str.c_str();

	// finaly add the new row into result set. SYNCHRONIZED?
	M4DDicomServiceProvider::ResultSet *rs = 
		static_cast<M4DDicomServiceProvider::ResultSet *>(callbackData);

	rs->push_back(*row);
}