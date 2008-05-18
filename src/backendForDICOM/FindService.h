#ifndef M4DDICOMFINDSERVICE
#define M4DDICOMFINDSERVICE

/**
 *  Implements C-FIND service to DICOM server
 */
#include <string>
#include "dicomConn/DICOMServiceProvider.h"

namespace M4D
{
namespace DicomInternal {

class FindService : AbstractService
{
private :
	DcmDataset *queryForFilterTable;

	void GetQuery( 
		DcmDataset **query, 
		const string *patientName,
		const string *patientID,
		const string *modality,
		const string *date );

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
		DIMSE_FindUserCallback callBack) ;

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
		Dicom::DcmProvider::ResultSet &result,
		const string &patientName,
		const string &patientID,
		const Dicom::DcmProvider::StringVector &modalities,
		const string &dateFrom,
		const string &dateTo) ;

	void FindStudiesAboutPatient(
		const string &patientID,
		Dicom::DcmProvider::ResultSet &result) ;

	void FindWholeStudyInfo(
		const string &patientID,
		const string &studyID,
		Dicom::DcmProvider::StudyInfo &info) ;

	void FindStudyInfo(
		const string &patientID,
		const string &studyID,
		Dicom::DcmProvider::StringVector &seriesIDs) ; 

	
};

} // namespace
}

#endif

