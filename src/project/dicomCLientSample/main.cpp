
#include "Debug.h"
#include "Log.h"
#include "Common.h"
#include "ExceptionBase.h"
#include "dicomConn/M4DDICOMServiceProvider.h"

using namespace std;
using namespace M4D::Dicom;

int
main( void)
{
	DcmProvider::ResultSet result;
	DcmProvider::StudyInfo studyInfo;
	DcmProvider::DicomObjSet obj;	

	std::string patName = "";
	std::string patID = "";
	DcmProvider::StringVector modalities;
	std::string dateFrom;
	std::string dateTo;

	try {
		// provider object
		DcmProvider provider;

		// find some patient & studies info
		provider.Find(
		result,
		patName,
		patID,
		modalities,
		dateFrom,
		dateTo);							

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



	} catch( M4D::ErrorHandling::ExceptionBase &e) {
		string what = e.what();
		return -1;
	}

    return 0;
}
