#ifndef M4D_DICOM_SERVICE_PROVIDER
#define M4D_DICOM_SERVICE_PROVIDER

#include <vector>
#include <map>

using namespace std;

#include "M4DDICOMObject.h"

class M4DDicomServiceProvider
{
public:
	// TYPEDEFS ///////////////////////////////////////////////////////////

	// represents one row in table that shows found results
	typedef struct s_browserRow {
		string patientName;
		bool patientSex;
		string patientBirthDate;
		string patentID;
		string studyID;
		string studyDate;
		string modality;
	} M4DTableRow;

	typedef vector<M4DTableRow> ResultSet;
	typedef vector<string> StringVector;

	// vector of image IDs
	typedef vector<string> M4DImageIDsInSet;
	// vector of M4DSetInfo
	typedef map<string, M4DImageIDsInSet> M4DStudyInfo;

	typedef vector<M4DDicomObj> M4DDicomObjSet;

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
		M4DStudyInfo &info);

	void GetImage(
		string &patientID,
		string &studyID,
		string &serieID,
		string &imageID,
		M4DDicomObj &object);

	void GetImageSet(
		string &patientID,
		string &studyID,
		string &serieID,
		M4DDicomObjSet &result);
};

#endif