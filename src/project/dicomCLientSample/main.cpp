
#include "Debug.h"
#include "Log.h"
#include "M4DCommon.h"
#include "dicomConn/M4DDICOMServiceProvider.h"

using namespace std;
using namespace M4DDicom;

int
main( void)
{
	DcmProvider::ResultSet result;
	DcmProvider::StudyInfo studyInfo;
	DcmProvider::DicomObjSet obj;	

	string patientName = "";
	string patientID = "";
	string modality = "";
	string dateFrom = "";

	try {
		// provider object
		DcmProvider provider;

		// find some patient & studies info
		provider.Find( 
			result, patientName, patientID, modality, dateFrom);

		DcmProvider::TableRow *row = &result[0];

		// find some info about selected study
		provider.WholeFindStudyInfo( row->patentID, row->studyID, studyInfo);

		// now get image
		provider.GetImageSet( row->patentID, row->studyID,
			studyInfo.begin()->first, obj);

		int i=0;
		char fileName[32];
		string pok;
		for( DcmProvider::DicomObjSet::iterator it = obj.begin();
			it != obj.end();
			it++)
		{
			//pok = it->GetPatientName();
			sprintf( fileName, "C:\\pok%d.dcm", i++);
			it->Save( fileName );
		}

		//M4DDcmProvider::DicomObj obj;



	} catch( bad_exception *e) {
		string what = e->what();
		delete e;
		return -1;
	}

    return 0;
}