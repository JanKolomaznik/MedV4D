
#include "main.h"
#include "M4DDICOMServiceProvider.h"

#ifdef WITH_OPENSSL
#include "dcmtk/dcmtls/tlstrans.h"
#include "dcmtk/dcmtls/tlslayer.h"
#endif

using namespace std;

int
main( void)
{
	M4DDcmProvider::ResultSet result;
	M4DDcmProvider::StudyInfo studyInfo;
	M4DDcmProvider::DicomObjSet obj;	

	string patientName = "";
	string patientID = "";
	string modality = "";
	string dateFrom = "";

	try {
		// provider object
		M4DDcmProvider provider;

		// find some patient & studies info
		provider.Find( 
			result, patientName, patientID, modality, dateFrom);

		M4DDcmProvider::TableRow *row = &result[0];

		// find some info about selected study
		provider.WholeFindStudyInfo( row->patentID, row->studyID, studyInfo);

		// now get image
		provider.GetImageSet( row->patentID, row->studyID,
			studyInfo.begin()->first, obj);

		int i=0;
		char fileName[32];
		string pok;
		for( M4DDcmProvider::DicomObjSet::iterator it = obj.begin();
			it != obj.end();
			it++)
		{
			//pok = it->GetPatientName();
			sprintf( fileName, "C:\\pok%d.dcm", i++);
			it->Save( fileName );
		}

		//M4DDcmProvider::DicomObj obj;



	} catch( bad_exception *e) {
		DOUT << e->what();
		delete e;
		return -1;
	}

    return 0;
}



