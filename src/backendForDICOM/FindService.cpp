#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"
#include "dcmtk/dcmdata/dcdeftag.h"

#include "M4DDicomAssoc.h"
#include "M4DDICOMServiceProvider.h"
#include "AbstractService.h"
#include "FindService.h"


///////////////////////////////////////////////////////////////////////

/**
 *  Creates DICOM data set containing all neccessary tags for query the server
 */
void
M4DFindService::GetQuery( 
	DcmDataset **query, 
	string *patientName,
	string *patientID,
	string *modality,
	string *dateFrom,
	string *dateTo
	)
{
	if (*query != NULL) delete *query;
    *query = new DcmDataset;

    DU_putStringDOElement(*query, DCM_QueryRetrieveLevel, "STUDY");
	// patient info
	DU_putStringDOElement(*query, DCM_PatientID, patientID->c_str() );
	DU_putStringDOElement(*query, DCM_PatientsName, patientName->c_str() );
	DU_putStringDOElement(*query, DCM_PatientsSex, NULL);
	DU_putStringDOElement(*query, DCM_PatientsBirthDate, NULL);

	// study info
    DU_putStringDOElement(*query, DCM_StudyInstanceUID, NULL);
	DU_putStringDOElement(*query, DCM_StudyDate, dateFrom->c_str() );
	DU_putStringDOElement(*query, DCM_Modality, modality->c_str() );

	// serie info
    //DU_putStringDOElement(*query, DCM_SeriesInstanceUID, NULL);	
    //DU_putStringDOElement(*query, DCM_SeriesNumber, NULL);
}

///////////////////////////////////////////////////////////////////////

void
M4DFindService::GetStudyInfoQuery( 
	DcmDataset **query, 
	string &patientID,
	string &studyID)
{
	if (*query != NULL) delete *query;
    *query = new DcmDataset;

    DU_putStringDOElement(*query, DCM_QueryRetrieveLevel, "IMAGES");
	// patient info
	DU_putStringDOElement(*query, DCM_PatientID, patientID.c_str() );

	// study info
	DU_putStringDOElement(*query, DCM_StudyInstanceUID, studyID.c_str());

	// serie info
    DU_putStringDOElement(*query, DCM_SeriesInstanceUID, NULL);

	// image info
	DU_putStringDOElement(*query, DCM_SOPInstanceUID, NULL);
}

///////////////////////////////////////////////////////////////////////

M4DFindService::M4DFindService()
{
	/* initialize network, i.e. create an instance of T_ASC_Network*. */
	#define TIME_OUT 30

	OFCondition cond = ASC_initializeNetwork(
		NET_REQUESTOR, 0, TIME_OUT, &m_net);
	if (cond.bad()) {
		DimseCondition::dump(cond);
		exit(1);
	}

	// create assotiation
	m_assocToServer = new M4DDicomAssociation( "find");
}

///////////////////////////////////////////////////////////////////////

M4DFindService::~M4DFindService()
{
	delete m_assocToServer;
}

///////////////////////////////////////////////////////////////////////

void
M4DFindService::FindSupport( 
	DcmDataset &queryDataSet,
	void *data,
	DIMSE_FindUserCallback callBack) throw (...)
{
	// request assoc to server
	m_assocToServer->Request( m_net);

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

    /* if required, dump some more general information */
    //if (opt_verbose) {
        printf("Find SCU RQ: MsgID %d\n", msgId);
        printf("REQUEST:\n");
        queryDataSet.print(COUT);
        printf("--------\n");
    //}

#define FIND_OPER_TIMEOUT 0

    /* finally conduct transmission of data */
	OFCondition cond = DIMSE_findUser(
		m_assocToServer->GetAssociation(),
		presId, &req, &queryDataSet,
		callBack, data,
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
        queryDataSet.print(COUT);
        DimseCondition::dump(cond);
    }

    /* dump status detail information if there is some */
    if (statusDetail != NULL) {
        printf("  Status Detail:\n");
        statusDetail->print(COUT);
        delete statusDetail;
    }

	m_assocToServer->Release();
}

///////////////////////////////////////////////////////////////////////

void
M4DFindService::FindStudiesAboutPatient( 
		M4DDicomServiceProvider::ResultSet &result, 
		string &patientID) throw (...)
{
	// create query
	DcmDataset *query = NULL;
	GetQuery( &query, NULL, &patientID, NULL, NULL, NULL);

	// issue it
	FindSupport( *query, (void *)&result, M4DFindService::TableRowCallback);
}

///////////////////////////////////////////////////////////////////////

void
M4DFindService::FindForFilter( 
		M4DDicomServiceProvider::ResultSet &result, 
		string &patientName,
		string &patientID,
		string &modality,
		string &dateFrom,
		string &dateTo) throw (...)
{
	// create query
	DcmDataset *query = NULL;
	GetQuery( &query, &patientName, &patientID, &modality, &dateFrom, &dateTo);

	// issue
	FindSupport( *query, (void *)&result, M4DFindService::TableRowCallback);
}

///////////////////////////////////////////////////////////////////////

void
M4DFindService::FindStudyInfo(
		string &patientID,
		string &studyID,
		M4DDicomServiceProvider::M4DStudyInfo &info) throw (...)
{
	// create query
	DcmDataset *query = NULL;
	GetStudyInfoQuery( &query, patientID, studyID);

	// issue
	FindSupport( *query, (void *)&info, M4DFindService::StudyInfoCallback);
}

///////////////////////////////////////////////////////////////////////

void
M4DFindService::TableRowCallback(
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
	responseIdentifiers->findAndGetOFString( DCM_StudyInstanceUID, str);
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

///////////////////////////////////////////////////////////////////////

void
M4DFindService::StudyInfoCallback(
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
	OFString str;
	string imageID;
	string setID;

	// Parse the response
	responseIdentifiers->findAndGetOFString( DCM_SOPInstanceUID, str);
	imageID = str.c_str();
	responseIdentifiers->findAndGetOFString( DCM_SeriesInstanceUID, str);
	setID = str.c_str();

	// get container that recieved values should go into
	M4DDicomServiceProvider::M4DStudyInfo *setInfo = 
		static_cast<M4DDicomServiceProvider::M4DStudyInfo*>(callbackData);

	M4DDicomServiceProvider::M4DImageIDsInSet *setImages;
	// try to find if there is already just recieved setID within the container
	M4DDicomServiceProvider::M4DStudyInfo::iterator it = 
		setInfo->find( setID);
	
	if( it == setInfo->end() )
	{
		// create new M4DImageIDsInSet & insert it into setInfo
		M4DDicomServiceProvider::M4DImageIDsInSet buddy;
		setInfo->insert( 
			M4DDicomServiceProvider::M4DStudyInfo::value_type( setID, buddy) );
		setImages = &setInfo->find( setID)->second;
	}	
	else
	{
		setImages = &it->second;
	}

	// insert imageID
	setImages->push_back( imageID);
}