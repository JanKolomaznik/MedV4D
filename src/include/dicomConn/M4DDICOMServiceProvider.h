#ifndef M4D_DICOM_SERVICE_PROVIDER
#define M4D_DICOM_SERVICE_PROVIDER

#include <vector>

using namespace std;

#include "M4DDICOMObject.h"

class M4DDicomServiceProvider
{
public:
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

	void Find( 
		string &patientName,
		StringVector &modalities,
		ResultSet &result);

	void GetImage(
		string &patientID,
		string &studyID,
		string &serieID,
		string &imageID,
		M4DDicomObj &object);

	void GetImageSet(
		string &patientID,
		string &studyID,
		string &serieID
		/* dcmDataset &result */);
};

#endif