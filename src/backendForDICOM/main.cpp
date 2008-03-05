

#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"
#include "dcmtk/dcmdata/dcdeftag.h"

#include "main.h"

#include "M4DDicomAssoc.h"
#include "M4DDICOMServiceProvider.h"
#include "AbstractService.h"
#include "MoveService.h"
#include "FindService.h"

#ifdef WITH_OPENSSL
#include "dcmtk/dcmtls/tlstrans.h"
#include "dcmtk/dcmtls/tlslayer.h"
#endif

using namespace std;

// set log & debug streams
std::ostream *logStream = &std::cout;
std::ostream *pdout = &std::cout;

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
		// create service objects
		M4DFindService findService;

		// find some patient & studies info
		findService.FindForFilter( 
			result, patientName, patientID, modality, dateFrom);

		M4DDcmProvider::TableRow *row = &result[0];

		// find some info about selected study
		findService.FindStudyInfo( row->patentID, row->studyID, studyInfo);

		M4DMoveService moveService;
		// now get image
		moveService.MoveImageSet( row->patentID, row->studyID,
			studyInfo.begin()->first, obj);

		int i=0;
		char fileName[32];
		string pok;
		for( M4DDcmProvider::DicomObjSet::iterator it = obj.begin();
			it != obj.end();
			it++)
		{
			pok = it->GetPatientName();
			sprintf( fileName, "C:\\pok%d.dcm", i++);
			it->Save( fileName );
		}

		//M4DDcmProvider::DicomObj obj;



	} catch( bad_exception *e) {
		cout << e->what();
		delete e;
		return -1;
	}

    return 0;
}



