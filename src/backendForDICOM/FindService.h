#ifndef M4DDICOMFINDSERVICE
#define M4DDICOMFINDSERVICE

/**
 *  Implements C-FIND service to DICOM server
 */

using namespace std;

class M4DFindService : M4DAbstractService
{
private :
	DcmDataset *queryForFilterTable;

	void GetQuery( 
		DcmDataset **query, 
		const string *patientName,
		const string *patientID,
		const string *modality,
		const string *dateFrom );

	void GetStudyInfoQuery( 
		DcmDataset **query,
		const string &patientID,
		const string &studyID);

	void GetWholeStudyInfoQuery( 
		DcmDataset **query,
		const string &patientID,
		const string &studyID);

	void FindSupport( DcmDataset &queryDataSet,
		void *data,
		DIMSE_FindUserCallback callBack) throw (...);

	static void
	TableRowCallback(
        void *callbackData,
        T_DIMSE_C_FindRQ *request,
        int responseCount,
        T_DIMSE_C_FindRSP *rsp,
        DcmDataset *responseIdentifiers );

	static void
	WholeStudyInfoCallback(
        void *callbackData,
        T_DIMSE_C_FindRQ *request,
        int responseCount,
        T_DIMSE_C_FindRSP *rsp,
        DcmDataset *responseIdentifiers );	

	static void
	StudyInfoCallback(
        void *callbackData,
        T_DIMSE_C_FindRQ *request,
        int responseCount,
        T_DIMSE_C_FindRSP *rsp,
        DcmDataset *responseIdentifiers );	

public:
	M4DFindService();
	~M4DFindService();

	void FindForFilter( 
		M4DDcmProvider::ResultSet &result,
		const string &patientName,
		const string &patientID,
		const string &modality,
		const string &date) throw (...);

	void FindStudiesAboutPatient(
		const string &patientID,
		M4DDcmProvider::ResultSet &result) throw (...);

	void FindWholeStudyInfo(
		const string &patientID,
		const string &studyID,
		M4DDcmProvider::StudyInfo &info) throw (...);

	void FindStudyInfo(
		const string &patientID,
		const string &studyID,
		M4DDcmProvider::StringVector &seriesIDs) throw (...); 

	
};

#endif