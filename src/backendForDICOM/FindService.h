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
		string *patientName,
		string *patientID,
		string *modality,
		string *dateFrom,
		string *dateTo );

	void GetStudyInfoQuery( 
		DcmDataset **query,
		string &patientID,
		string &studyID);

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
		M4DDicomServiceProvider::ResultSet &result,
		string &patientName,
		string &patientID,
		string &modality,
		string &dateFrom,
		string &dateTo) throw (...);

	void FindStudiesAboutPatient( 
		M4DDicomServiceProvider::ResultSet &result, 
		string &patientID) throw (...);

	void FindStudyInfo(
		string &patientID,
		string &studyID,
		M4DDicomServiceProvider::M4DStudyInfo &info) throw (...);

	
};

#endif