#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"
#include "dcmtk/dcmdata/dcdeftag.h"

#include "main.h"

#include "M4DDicomAssoc.h"
#include "M4DDICOMServiceProvider.h"
#include "AbstractService.h"
#include "FindService.h"

namespace M4D
{
using namespace Dicom;

namespace DicomInternal {

///////////////////////////////////////////////////////////////////////

/**
 *  Creates DICOM data set containing all neccessary tags for query the server
 */
void
FindService::GetQuery( 
	DcmDataset **query, 
	const string *patientName,
	const string *patientID,
	const string *modality,
	const string *date
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
	DU_putStringDOElement(*query, DCM_StudyDate, date->c_str() );
	DU_putStringDOElement(*query, DCM_Modality, modality->c_str() );

	// serie info
    //DU_putStringDOElement(*query, DCM_SeriesInstanceUID, NULL);	
    //DU_putStringDOElement(*query, DCM_SeriesNumber, NULL);
}

///////////////////////////////////////////////////////////////////////

void
FindService::GetWholeStudyInfoQuery( 
	DcmDataset **query, 
	const string &patientID,
	const string &studyID)
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

void
FindService::GetStudyInfoQuery( 
	DcmDataset **query, 
	const string &patientID,
	const string &studyID)
{
	if (*query != NULL) delete *query;
    *query = new DcmDataset;

    DU_putStringDOElement(*query, DCM_QueryRetrieveLevel, "SERIES");
	// patient info
	DU_putStringDOElement(*query, DCM_PatientID, patientID.c_str() );

	// study info
	DU_putStringDOElement(*query, DCM_StudyInstanceUID, studyID.c_str());

	// serie info
    DU_putStringDOElement(*query, DCM_SeriesInstanceUID, NULL);
}

///////////////////////////////////////////////////////////////////////

FindService::FindService()
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

FindService::~FindService()
{
	delete m_assocToServer;
}

///////////////////////////////////////////////////////////////////////

void
FindService::FindSupport( 
	DcmDataset &queryDataSet,
	void *data,
	DIMSE_FindUserCallback callBack) 
{
	// request assoc to server
	m_assocToServer->Request( m_net);

	DIC_US msgId = m_assocToServer->GetAssociation()->nextMsgID++;
    T_DIMSE_C_FindRQ req;
    T_DIMSE_C_FindRSP rsp;
    DcmDataset *statusDetail = NULL;

    /* figure out which of the accepted presentation contexts should be used */
	T_ASC_PresentationContextID presId = m_assocToServer->FindPresentationCtx();

    /* prepare the transmission of data */
    bzero((char*)&req, sizeof(req));
    req.MessageID = msgId;
	strcpy(req.AffectedSOPClassUID, 
		m_assocToServer->GetAssocAddress()->transferModel.c_str() );
    req.DataSetType = DIMSE_DATASET_PRESENT;
    req.Priority = DIMSE_PRIORITY_LOW;

    /* prepare the callback data */
	//MyCallbackInfo callbackData;
    //callbackData.assoc = assoc;
    //callbackData.presId = presId;

	LOG( "Find-SCU REQUEST");
    queryDataSet.print(LOUT);

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
		LOG( "Response: " << DU_cfindStatusString(rsp.DimseStatus));
    } else {
        D_PRINT("Find Failed, query keys:");
        queryDataSet.print(LOUT);
    }

    /* dump status detail information if there is some */
    if (statusDetail != NULL) {
        D_PRINT("Status Detail:");
        statusDetail->print(LOUT);
        delete statusDetail;
    }

	m_assocToServer->Release();
}

///////////////////////////////////////////////////////////////////////

void
FindService::FindStudiesAboutPatient( 
		const string &patientID,
		DcmProvider::ResultSet &result) 
{
	// create query
	DcmDataset *query = NULL;
	GetQuery( &query, NULL, &patientID, NULL, NULL);

	// issue it
	FindSupport( *query, (void *)&result, FindService::TableRowCallback);
}

///////////////////////////////////////////////////////////////////////

void
FindService::FindForFilter( 
		DcmProvider::ResultSet &result, 
		const string &patientName,
		const string &patientID,
		const string &modality,
		const string &date) 
{
	// create query
	DcmDataset *query = NULL;
	GetQuery( &query, &patientName, &patientID, &modality, &date);

	// issue
	FindSupport( *query, (void *)&result, FindService::TableRowCallback);
}

///////////////////////////////////////////////////////////////////////

void
FindService::FindWholeStudyInfo(
		const string &patientID,
		const string &studyID,
		DcmProvider::StudyInfo &info) 
{
	// create query
	DcmDataset *query = NULL;
	GetWholeStudyInfoQuery( &query, patientID, studyID);

	// issue
	FindSupport( *query, (void *)&info, FindService::WholeStudyInfoCallback);
}

///////////////////////////////////////////////////////////////////////

void
FindService::FindStudyInfo(
		const string &patientID,
		const string &studyID,
		DcmProvider::StringVector &seriesIDs) 
{
	// create query
	DcmDataset *query = NULL;
	GetStudyInfoQuery( &query, patientID, studyID);

	// issue
	FindSupport( *query, (void *)&seriesIDs, FindService::StudyInfoCallback);
}

///////////////////////////////////////////////////////////////////////

void
FindService::TableRowCallback(
        void *callbackData,
        T_DIMSE_C_FindRQ * /*request*/,
        int /*responseCount*/,
        T_DIMSE_C_FindRSP * /*rsp*/,
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

	//////////////////////////////////////////////////
	// Parse the response
	//////////////////////////////////////////////////
	DcmProvider::TableRow *row = new DcmProvider::TableRow;

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
	DcmProvider::ResultSet *rs = 
		static_cast<DcmProvider::ResultSet *>(callbackData);

	rs->push_back(*row);
}

///////////////////////////////////////////////////////////////////////

void
FindService::WholeStudyInfoCallback(
        void *callbackData,
        T_DIMSE_C_FindRQ * /*request*/,
        int /*responseCount*/,
        T_DIMSE_C_FindRSP * /*rsp*/,
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
	DcmProvider::StudyInfo *setInfo = 
		static_cast<DcmProvider::StudyInfo*>(callbackData);

	DcmProvider::StringVector *setImages;
	// try to find if there is already just recieved setID within the container
	DcmProvider::StudyInfo::iterator it = 
		setInfo->find( setID);
	
	if( it == setInfo->end() )
	{
		// create new StringVector & insert it into setInfo
		DcmProvider::StringVector buddy;
		setInfo->insert( 
			DcmProvider::StudyInfo::value_type( setID, buddy) );
		setImages = &setInfo->find( setID)->second;
	}	
	else
	{
		setImages = &it->second;
	}

	// insert imageID
	setImages->push_back( imageID);
}

///////////////////////////////////////////////////////////////////////

void
FindService::StudyInfoCallback(
        void *callbackData,
        T_DIMSE_C_FindRQ * /*request*/,
        int /*responseCount*/,
        T_DIMSE_C_FindRSP * /*rsp*/,
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
	string setID;

	// Parse the response
	responseIdentifiers->findAndGetOFString( DCM_SeriesInstanceUID, str);
	setID = str.c_str();

	// get container that recieved values should go into
	DcmProvider::StringVector *setInfo = 
		static_cast<DcmProvider::StringVector *>(callbackData);

	setInfo->push_back( setID);
}

} // namespace
}