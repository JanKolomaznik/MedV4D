
//#include "dcmtk/config/osconfig.h" /* make sure OS specific configuration is included first */

//#define INCLUDE_CSTDLIB
//#define INCLUDE_CSTDIO
//#define INCLUDE_CSTRING
//#define INCLUDE_CSTDARG
//#define INCLUDE_CERRNO
//#include "dcmtk/ofstd/ofstdinc.h"

//BEGIN_EXTERN_C
//#ifdef HAVE_SYS_FILE_H
//#include <sys/file.h>
//#endif
//END_EXTERN_C

#ifdef HAVE_GUSI_H
#include <GUSI.h>
#endif

//#include "dcmtk/dcmnet/dimse.h"
//#include "dcmtk/dcmnet/diutil.h"
//#include "dcmtk/dcmdata/dcfilefo.h"
//#include "dcmtk/dcmdata/dcdebug.h"
//#include "dcmtk/dcmdata/dcuid.h"
//#include "dcmtk/dcmdata/dcdict.h"
//#include "dcmtk/dcmdata/cmdlnarg.h"
//#include "dcmtk/ofstd/ofconapp.h"
//#include "dcmtk/dcmdata/dcuid.h"    /* for dcmtk version name */
//#include "dcmtk/dcmdata/dcdicent.h"
//#include "dcmtk/dcmdata/dcdeftag.h"

#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"
#include "dcmtk/dcmdata/dcdeftag.h"

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

int
main( void)
{
	M4DDicomServiceProvider::ResultSet result;
	M4DDicomServiceProvider::M4DStudyInfo studyInfo;
	M4DDicomServiceProvider::M4DDicomObjSet obj;

	string patientName = "";
	string patientID = "";
	string modality = "";
	string dateFrom = "";
	string dateTo = "";

	try {
		// create service objects
		M4DFindService findService;

		// find some patient & studies info
		findService.FindForFilter( 
			result, patientName, patientID, modality, dateFrom, dateTo);

		M4DDicomServiceProvider::M4DTableRow *row = &result[0];

		// find some info about selected study
		findService.FindStudyInfo( row->patentID, row->studyID, studyInfo);

		M4DMoveService moveService;
		// now get image
		moveService.MoveImageSet( row->patentID, row->studyID,
			studyInfo.begin()->first, obj);

		int i=0;
		char fileName[32];
		for( M4DDicomServiceProvider::M4DDicomObjSet::iterator it = obj.begin();
			it != obj.end();
			it++)
		{
			sprintf( fileName, "C:\\pok%d.dcm", i++);
			it->Save( fileName );
		} 


	} catch( bad_exception *e) {
		cout << e->what();
		delete e;
		return -1;
	}

    return 0;
}



