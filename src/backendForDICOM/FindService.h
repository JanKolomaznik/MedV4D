#ifndef M4DDICOMFINDSERVICE
#define M4DDICOMFINDSERVICE

/**
 *  Implements C-FIND service to DICOM server. 
 *  Process decription in a nutshell: client (SCU) establish assotiation to server (SCP) and sends query dataSet. Server process query dataSet and sends back matched results.
 *  For more details see DICOM doc ([ver]_08.pdf chapter 9.1.2) and coresponding annexes).
 */
#include <string>

using namespace M4D::Dicom;

namespace M4D
{
namespace DicomInternal 
{

class FindService : AbstractService
{
  friend class M4D::Dicom::DcmProvider;

	DcmDataset *queryForFilterTable;

  // qeury dataSet preparation function. Each for each retrieve function
	void GetQuery( 
		DcmDataset **query, 
		const string *patientName,
		const string *patientID,
	  const string *date,
    const string *referringMD,
    const string *description );
  // ...
	void GetStudyInfoQuery( 
		DcmDataset **query,
		const string &patientID,
		const string &studyID);
  // ...
	void GetWholeStudyInfoQuery( 
		DcmDataset **query,
		const string &patientID,
		const string &studyID);

  // supporting funcion common for all retrival functions
	void FindSupport( DcmDataset &queryDataSet,
		void *data,
		DIMSE_FindUserCallback callBack) ;

  // Supporting callbacks that cooperates with DCMTK functions ...
	static void
	TableRowCallback(
        void *callbackData,
        T_DIMSE_C_FindRQ *request,
        int responseCount,
        T_DIMSE_C_FindRSP *rsp,
        DcmDataset *responseIdentifiers );
  // ...
	static void
	WholeStudyInfoCallback(
        void *callbackData,
        T_DIMSE_C_FindRQ *request,
        int responseCount,
        T_DIMSE_C_FindRSP *rsp,
        DcmDataset *responseIdentifiers );	
  // ...
	static void
	StudyInfoCallback(
        void *callbackData,
        T_DIMSE_C_FindRQ *request,
        int responseCount,
        T_DIMSE_C_FindRSP *rsp,
        DcmDataset *responseIdentifiers );	

  // ctor & dtor
	FindService();
	~FindService();

  /**
   *  Send find request to server with appropriate query dataSet based on parameters of this functions (filter).
   *  Returns resultSet container with table rows records.
   */
	void FindForFilter( 
		Dicom::DcmProvider::ResultSet &result,
		const string &patientForeName,
    const string &patientSureName,
    const string &patID,
		const string &dateFrom,
		const string &dateTo,
    const string &referringMD,
    const string &description);

  /**
   *  Send find request to server with appropriate query dataSet based on parameters of this functions (filter).
   *  Returns resultSet container with table rows records.
   */
	void FindStudiesAboutPatient(
		const string &patientID,
		Dicom::DcmProvider::ResultSet &result);

  /**
   *  Send find request to server, returns study info (map keyed by IDs of series and valued by IDs of images of appropriate serie)
   *  Used when user selects record in table. Because each table record is on study info it is neccessary to retrieve info of lower level than study (serie, images). In normal case there is one serie in study. But can be more.
   */
	void FindWholeStudyInfo(
		const string &patientID,
		const string &studyID,
		Dicom::DcmProvider::StudyInfo &info);

  /**
   *  The same as FindWholeStudyInfo, but find only serie IDs. Used when whole serie images are to be retrive later on
   */
	void FindStudyInfo(
		const string &patientID,
		const string &studyID,
    Dicom::DcmProvider::SerieInfoVector &seriesIDs); 

	
};

} // namespace
}

#endif

