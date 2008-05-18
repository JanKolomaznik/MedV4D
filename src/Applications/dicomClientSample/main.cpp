
#include "Common.h"
#include "ExceptionBase.h"
#include "dicomConn/DICOMServiceProvider.h"

using namespace std;
using namespace M4D::Dicom;

const unsigned FILENAME_BUFFER_SIZE = 32;

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
		char fileName[FILENAME_BUFFER_SIZE*8];
		string pok;
		for( DcmProvider::DicomObjSet::iterator it = obj.begin();
			it != obj.end();
			it++)
		{
			//pok = it->GetPatientName();
			sprintf_s( fileName, FILENAME_BUFFER_SIZE*8, "C:\\dicomServer\\recievedImages\\pok%d.dcm", i++);
			it->Save( fileName );
  
		}

    //float thickness;
    //float vertSpacing, horizSpacing;
    //float slicePosition;

    //DcmProvider::DicomObj obj;

    //provider.GetImage(std::string("XXXXXXXX"),
    //  std::string("1.3.12.2.1107.5.1.4.57132.30000006100305415862500000037"),
    //  std::string("1.3.12.2.1107.5.1.4.57132.30000006100305432756200002333"),
    //  std::string("1.3.12.2.1107.5.1.4.57132.30000006100305432756200002337"),
    //  obj);

    //obj.GetSliceThickness( thickness);
    //obj.GetSliceLocation( slicePosition);
    //obj.GetPixelSpacing( horizSpacing, vertSpacing);

    //uint16 *arr = new uint16[obj.GetWidth()*obj.GetHeight()];
    //obj.FlushIntoArray<uint16>(arr);
		//M4DDcmProvider::DicomObj obj;



	} catch( M4D::ErrorHandling::ExceptionBase &e) {
		string what = e.what();
		return -1;
	}

    return 0;
}
