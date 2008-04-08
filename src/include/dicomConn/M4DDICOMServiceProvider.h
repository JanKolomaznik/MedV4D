#ifndef M4D_DICOM_SERVICE_PROVIDER
#define M4D_DICOM_SERVICE_PROVIDER

#include <vector>
#include <map>

using namespace std;

class M4DDcmProvider
{
private:
	void *findService;
	void *moveService;
public:

	// TYPEDEFS ///////////////////////////////////////////////////////////

	class DicomObj;

	// represents one row in table that shows found results
	typedef struct s_browserRow {
		string patientName;
		bool patientSex;
		string patientBirthDate;
		string patentID;
		string studyID;
		string studyDate;
		string modality;
	} TableRow;

	typedef vector<TableRow> ResultSet;

	// vector of image IDs
	typedef vector<string> StringVector;
	// vector of M4DSetInfo
	typedef map<string, StringVector> StudyInfo;

	typedef vector<DicomObj> DicomObjSet;

	// METHODs ////////////////////////////////////////////////////////////

	// user inputs some values into seach form & wants a table to be filled
	// with appropriate results.
	/**
	 *	@param result - reference to result set object
	 *	@param patientName - reference to string containing ...
	 *	@param patientID   -   - || -
	 *	@param modalities  - reference to string containing set 
	 *		of wanted modalities divided by '\' character
	 *	@param date		   - ref. to string containing one of theese variants:
	 *		<dateFrom>- = all items that has date >= than 'dateFrom'
	 *		-<dateTo>	= all items thats' date <= than 'dateTo'
	 *		<dateFrom>-<dateTo> = all items between that dates
	 *		<dateX> = string with format YYYYMMDD
	 */
	void Find( 
		M4DDcmProvider::ResultSet &result,
		const string &patientName,
		const string &patientID,
		const string &modalities,
		const string &date) throw (...);

	// user select any study and wants to get seriesInstanceUIDs
	void FindStudyInfo(
		const string &patientID,
		const string &studyID,
		StringVector &info) throw (...);

	// the same as FindStudyInfo but gets even imageIDs
	void WholeFindStudyInfo(
		const string &patientID,
		const string &studyID,
		StudyInfo &info) throw (...);

	// finds all studies concerning given patient
	void FindStudiesAboutPatient(  
		const string &patientID,
		ResultSet &result) throw (...);

	void GetImage(
		const string &patientID,
		const string &studyID,
		const string &serieID,
		const string &imageID,
		DicomObj &object) throw (...);

	void GetImageSet(
		const string &patientID,
		const string &studyID,
		const string &serieID,
		DicomObjSet &result) throw (...);

	M4DDcmProvider();
	~M4DDcmProvider();
};

#include "M4DDICOMObject.h"

#endif