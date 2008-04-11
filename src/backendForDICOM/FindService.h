#ifndef M4DDICOMFINDSERVICE
#define M4DDICOMFINDSERVICE

/**
 *  Implements C-FIND service to DICOM server
 */
#include <string>
using namespace std;

using namespace M4DDicom;

namespace M4DDicomInternal {

class FindService : AbstractService
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
	FindService();
	~FindService();

	void FindForFilter( 
		DcmProvider::ResultSet &result,
		const string &patientName,
		const string &patientID,
		const string &modality,
		const string &date) throw (...);

	void FindStudiesAboutPatient(
		const string &patientID,
		DcmProvider::ResultSet &result) throw (...);

	void FindWholeStudyInfo(
		const string &patientID,
		const string &studyID,
		DcmProvider::StudyInfo &info) throw (...);

	void FindStudyInfo(
		const string &patientID,
		const string &studyID,
		DcmProvider::StringVector &seriesIDs) throw (...); 

	
};

} // namespace

#endif