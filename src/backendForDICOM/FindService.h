#ifndef M4DDICOMFINDSERVICE
#define M4DDICOMFINDSERVICE

/**
 *  @ingroup dicom
 *  @file FindService.h
 *  @author Vaclav Klecanda
 */

/// Implements C-FIND service to DICOM server. 
/** Process decription in a nutshell: client (SCU) establish assotiation to server (SCP) and sends query dataSet. Server process query dataSet and sends back matched results.
 *  For more details see DICOM doc ([ver]_08.pdf chapter 9.1.2) and coresponding annexes).
 */
#include <string>
#include "structures.h"

namespace M4D {
namespace Dicom {

class FindService : AbstractService {
public:
	// ctor & dtor
	FindService();
	~FindService();

	/// Send find request searching for studies.
	/** to server with appropriate query dataSet based on parameters of this functions (filter).
	 *  Returns resultSet container with table rows records.
	 */
	void 
	FindForFilter(
			ResultSet &result, 
			const std::string &patientForeName,
			const std::string &patientSureName, 
			const std::string &patID,
			const std::string &dateFrom, 
			const std::string &dateTo,
			const std::string &referringMD, 
			const std::string &description
			);

	/// Send find request searching for patient's studies.
	/** to server with appropriate query dataSet based on parameters of this functions (filter).
	 *  Returns resultSet container with table rows records.
	 */
	void 
	FindStudiesAboutPatient(
			const std::string &patientID, 
			ResultSet &result
			);

	/// Send find request searching for study info.
	/** to server, returns study info (map keyed by IDs of series and valued by IDs of images of appropriate serie)
	 *  Used when user selects record in table. Because each table record is on study info it is neccessary to retrieve info of lower level than study (serie, images). In normal case there is one serie in study. But can be more.
	 */
	void 
	FindWholeStudyInfo(
			const std::string &patientID, 
			const std::string &studyID,
			StudyInfo &info
			);

	/**
	 *  The same as FindWholeStudyInfo, but find only serie IDs. Used when whole serie images are to be retrive later on
	 */
	void 
	FindStudyInfo(
			const std::string &patientID, 
			const std::string &studyID,
			SerieInfoVector &seriesIDs
			);
private:

	DcmDataset *queryForFilterTable;

	// qeury dataSet preparation function. Each for each retrieve function
	void GetQuery(
			DcmDataset **query, 
			const std::string *patientName,
			const std::string *patientID, 
			const std::string *date,
			const std::string *referringMD, 
			const std::string *description
			);
	// ...
	void GetStudyInfoQuery(
			DcmDataset **query, 
			const std::string &patientID,
			const std::string &studyID
			);
	// ...
	void GetWholeStudyInfoQuery(
			DcmDataset **query, 
			const std::string &patientID,
			const std::string &studyID
			);

	// supporting funcion common for all retrival functions
	void FindSupport(
			DcmDataset &queryDataSet, 
			void *data,
			DIMSE_FindUserCallback callBack
			);

	// Supporting callbacks that cooperates with DCMTK functions ...
	static void TableRowCallback(
			void *callbackData, 
			T_DIMSE_C_FindRQ *request,
			int responseCount, 
			T_DIMSE_C_FindRSP *rsp,
			DcmDataset *responseIdentifiers
			);
	// ...
	static void WholeStudyInfoCallback(
			void *callbackData,
			T_DIMSE_C_FindRQ *request, 
			int responseCount,
			T_DIMSE_C_FindRSP *rsp, 
			DcmDataset *responseIdentifiers
			);
	// ...
	static void StudyInfoCallback(
			void *callbackData,
			T_DIMSE_C_FindRQ *request, 
			int responseCount,
			T_DIMSE_C_FindRSP *rsp, 
			DcmDataset *responseIdentifiers
			);

};

} // namespace
}

/** @} */

#endif

