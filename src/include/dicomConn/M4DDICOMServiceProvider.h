#ifndef M4D_DICOM_SERVICE_PROVIDER
#define M4D_DICOM_SERVICE_PROVIDER

#include <vector>
#include <map>

using namespace std;

class M4DDcmProvider
{
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
	typedef vector<string> StringVector;

	// vector of image IDs
	typedef vector<string> ImageIDsInSet;
	// vector of M4DSetInfo
	typedef map<string, ImageIDsInSet> StudyInfo;

	typedef vector<DicomObj> M4DDicomObjSet;

	// METHODs ////////////////////////////////////////////////////////////

	// user inputs some values into seach form & wants a table to be filled
	// with appropriate results
	void Find( 
		string &patientName,
		StringVector &modalities,
		ResultSet &result);

	// user select any study and wants to get seriesInstanceUIDs & imageUDIs
	void FindStudyInfo(
		string &patientID,
		string &studyID,
		StudyInfo &info);

	void GetImage(
		string &patientID,
		string &studyID,
		string &serieID,
		string &imageID,
		DicomObj &object);

	void GetImageSet(
		string &patientID,
		string &studyID,
		string &serieID,
		M4DDicomObjSet &result);
};

#include "M4DDICOMObject.h"

#endif