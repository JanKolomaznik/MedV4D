#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"
#include "dcmtk/dcmdata/dcdeftag.h"

#include "main.h"

#include "M4DDicomAssoc.h"
#include "M4DDICOMServiceProvider.h"
#include "AbstractService.h"
#include "MoveService.h"
#include "FindService.h"

using namespace M4DDicomInternal;

// set log & debug streams
std::ostream *logStream = &std::cout;
std::ostream *pdout = &std::cout;

// create service objects

//StoreService storeService;

namespace M4DDicom {

///////////////////////////////////////////////////////////////////////

DcmProvider::DcmProvider()
{
	findService = new FindService();
	moveService = new MoveService();
}

DcmProvider::~DcmProvider()
{
	delete ( (FindService *) findService);
	delete ( (MoveService *) moveService);
}

///////////////////////////////////////////////////////////////////////
// Theese function are only inline redirections to member functions
// of appropriate service objects

void
DcmProvider::Find( 
		DcmProvider::ResultSet &result,
		const string &patientName,
		const string &patientID,
		const string &modalities,
		const string &date) throw (...)
{
	static_cast<FindService *>(findService)->FindForFilter( 
		result, patientName, patientID, modalities, date);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindStudyInfo(
		const string &patientID,
		const string &studyID,
		StringVector &info) throw (...)
{
	static_cast<FindService *>(findService)->FindStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::WholeFindStudyInfo(
		const string &patientID,
		const string &studyID,
		StudyInfo &info) throw (...)
{
	static_cast<FindService *>(findService)->FindWholeStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindStudiesAboutPatient(  
		const string &patientID,
		ResultSet &result) throw (...)
{
	static_cast<FindService *>(findService)->FindStudiesAboutPatient( 
		patientID, result);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::GetImage(
		const string &patientID,
		const string &studyID,
		const string &serieID,
		const string &imageID,
		DicomObj &object) throw (...)
{
	static_cast<MoveService *>(moveService)->MoveImage( 
		patientID, studyID, serieID, imageID, object);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::GetImageSet(
		const string &patientID,
		const string &studyID,
		const string &serieID,
		DicomObjSet &result) throw (...)
{
	static_cast<MoveService *>(moveService)->MoveImageSet( 
		patientID, studyID, serieID, result);
}

///////////////////////////////////////////////////////////////////////

} // namespace