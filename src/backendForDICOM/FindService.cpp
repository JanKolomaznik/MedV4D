/**
 *  @ingroup dicom
 *  @file FindService.cpp
 *  @author Vaclav Klecanda
 */
#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmnet/dimse.h>
#include <dcmtk/dcmnet/diutil.h>
#include <dcmtk/dcmdata/dcdeftag.h>

#include "MedV4D/Common/Common.h"

#include "../DicomAssoc.h"
#include "../AbstractService.h"
#include "../FindService.h"

#include "../DICOMSupport.h"

using namespace M4D::ErrorHandling;
using namespace M4D::Dicom;

///////////////////////////////////////////////////////////////////////

/**
 *  Creates DICOM data set containing all neccessary tags for query the server
 */
void
FindService::GetQuery( 
	DcmDataset **query, 
	const std::string *patientName,
  const std::string *patientID,
	const std::string *date,
  const std::string *referringMD,
  const std::string *description
	)
{
	if (*query != NULL) 
    delete *query;
  *query = new DcmDataset;

  /*
   *  Because there is no way to filter data based on patient/study 
   *  level info && series info (modality) so we have give up filtering
   *  according modality and this filtering do then on arrived results.
   */
  DU_putStringDOElement(*query, DCM_QueryRetrieveLevel, "STUDY");

	// patient info
  DU_putStringDOElement(*query, DCM_PatientID, patientID->c_str() );
	DU_putStringDOElement(*query, DCM_PatientName, patientName->c_str() );
	DU_putStringDOElement(*query, DCM_PatientSex, NULL);
	DU_putStringDOElement(*query, DCM_PatientBirthDate, NULL);

	// study info
  DU_putStringDOElement(*query, DCM_StudyInstanceUID, NULL);
	DU_putStringDOElement(*query, DCM_StudyDate, date->c_str() );
  DU_putStringDOElement(*query, DCM_Modality, NULL );
  DU_putStringDOElement(*query, DCM_StudyTime, NULL );
  DU_putStringDOElement(*query, DCM_ReferringPhysicianName, referringMD->c_str() );
  DU_putStringDOElement(*query, DCM_StudyDescription, description->c_str() );

}

///////////////////////////////////////////////////////////////////////

void
FindService::GetWholeStudyInfoQuery( 
	DcmDataset **query, 
	const std::string &patientID,
	const std::string &studyID)
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
	const std::string &patientID,
	const std::string &studyID)
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
	#define TIME_OUT 15

	OFCondition cond = ASC_initializeNetwork(
		NET_REQUESTOR, 0, TIME_OUT, &m_net);
	if (cond.bad()) {
		DimseCondition::dump(cond);
		exit(1);
	}

	// create assotiation
	m_assocToServer = new DicomAssociation( "find");
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
  try {
	  // request assoc to server
	  m_assocToServer->Request( m_net);
  } catch( ExceptionBase &e) {
    LOG( "C-FIND operation: " << e.what());
    throw;
  }

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

#ifdef _DEBUG
  D_PRINT( "Find-SCU REQUEST");
  D_PRINT( "-------------------");
  queryDataSet.print(LOUT);
#endif

#define FIND_OPER_TIMEOUT 0

    /* finally conduct transmission of data */
	OFCondition cond = DIMSE_findUser(
		m_assocToServer->GetAssociation(),
		presId, &req, &queryDataSet,
		callBack, data,
		m_mode, FIND_OPER_TIMEOUT,
		&rsp, &statusDetail);

  /* dump some more general information */
  if (cond == EC_Normal) 
  {
	  LOG( "Response: " << DU_cfindStatusString(rsp.DimseStatus));
  } else {
    LOG("Find Failed, query keys:");
    queryDataSet.print(LOUT);
  }

  /* dump status detail information if there is some */
  if (statusDetail != NULL) 
  {
#ifdef _DEBUG
    D_PRINT("Status Detail:");
    statusDetail->print(LOUT);
#endif
    
    delete statusDetail;
  }

	m_assocToServer->Release();
}

///////////////////////////////////////////////////////////////////////

void
FindService::FindStudiesAboutPatient( 
		const std::string &patientID,
		ResultSet &result) 
{
	// create query
	DcmDataset *query = NULL;
	GetQuery( &query, NULL, &patientID, NULL, NULL, NULL);

	// issue it
	FindSupport( *query, (void *)&result, FindService::TableRowCallback);
}

///////////////////////////////////////////////////////////////////////

void
FindService::FindForFilter( 
		ResultSet &result, 
		const std::string &patientForeName,
		const std::string &patientSureName,
		const std::string &patID,
		const std::string &dateFrom,
		const std::string &dateTo,
		const std::string &referringMD,
		const std::string &description
		) 
{
	// create range match date string. Range matching character '-'
	std::string dateRange;
	{
		std::stringstream dateStream;
		if( ! dateFrom.empty() && ! dateTo.empty() ) {
			dateStream << dateFrom << "-" << dateTo;
		} else if( ! dateFrom.empty() ) {
			dateStream << dateFrom << "-";
		} else if( ! dateTo.empty() ) {
			dateStream << "-" << dateTo;
		}
		dateRange = dateStream.str();
	}

	// create list matching for modalities. single items are delimited by '\'.
	// RETIRED !!
	//string modalityMatch;
	//{
	// stringstream modalitiesStram;
	// DcmProvider::StringVector::const_iterator it = modalities.begin();

	// if( it != modalities.end())
	// {
	//  modalitiesStram << *it;
	//  it++;
	// }

	// // append others
	// while( it != modalities.end())
	// {
	//  modalitiesStram << "\\" << *it;	
	//  it++;
	// }
	//  modalityMatch = modalitiesStram.str();
	//}

	// create wild card math for patient name
	std::string patienName;
	{
		std::stringstream s;
		if( ! patientForeName.empty()) {
			s << "*" << patientForeName << "*";
		}
		if( ! patientSureName.empty() ) {
			s << "*" << patientSureName << "*";
		}
		patienName = s.str();
	}

	// create wild card math for referring name
	std::string referringMDName;
	{
		std::stringstream s;
		if( ! referringMD.empty()) {
			s << "*" << referringMD << "*";
		}
		referringMDName = s.str();
	}

	// create wild card math for description
	std::string descr;
	{
		std::stringstream s;
		if( ! description.empty()) {
			s << "*" << description << "*";
		}
		descr = s.str();
	}

	// create query
	DcmDataset *query = NULL;
	GetQuery( &query, &patienName, &patID, &dateRange, &referringMDName, &descr);

	// issue
	FindSupport( *query, (void *)&result, FindService::TableRowCallback);
}

///////////////////////////////////////////////////////////////////////

void
FindService::FindWholeStudyInfo(
		const std::string &patientID,
		const std::string &studyID,
		StudyInfo &info
		) 
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
		const std::string &patientID,
		const std::string &studyID,
    		SerieInfoVector &seriesIDs
		) 
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
  TableRow *row = new TableRow;

	//////////////////////////////////////////////////
	// Parse the response
	//////////////////////////////////////////////////
	GetTableRowFromDataSet( responseIdentifiers, row);  

	// finaly add the new row into result set. SYNCHRONIZED?
	ResultSet *rs = 
		static_cast<ResultSet *>(callbackData);

	rs->push_back(*row);
  D_PRINT( "Next patient record arrived ...");
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
	std::string imageID;
	std::string setID;

	// Parse the response
	responseIdentifiers->findAndGetOFString( DCM_SOPInstanceUID, str);
	imageID = str.c_str();
	responseIdentifiers->findAndGetOFString( DCM_SeriesInstanceUID, str);
	setID = str.c_str();

	// get container that recieved values should go into
	StudyInfo *setInfo = 
		static_cast<StudyInfo*>(callbackData);

	StringVector *setImages;
	// try to find if there is already just recieved setID within the container
	StudyInfo::iterator it = 
		setInfo->find( setID);
	
	if( it == setInfo->end() )
	{
		// create new StringVector & insert it into setInfo
		StringVector buddy;
		setInfo->insert( 
			StudyInfo::value_type( setID, buddy) );
		setImages = &setInfo->find( setID)->second;
	}	
	else
	{
		setImages = &it->second;
	}

	// insert imageID
	setImages->push_back( imageID);
  D_PRINT( "Next record into study info arrived ...");
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
  SerieInfo seriesInfo;
  
  GetSeriesInfo( responseIdentifiers, &seriesInfo);

	// get container that recieved values should go into
  SerieInfoVector *setInfo = 
		static_cast<SerieInfoVector *>(callbackData);

	setInfo->push_back( seriesInfo);
  D_PRINT( "Next record into study info arrived ...");
}

/** @} */

