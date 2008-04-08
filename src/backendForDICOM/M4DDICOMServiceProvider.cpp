
#include "dcmtk/dcmnet/dimse.h"
#include "dcmtk/dcmnet/diutil.h"
#include "dcmtk/dcmdata/dcdeftag.h"

#include "main.h"

#include "M4DDicomAssoc.h"
#include "M4DDICOMServiceProvider.h"
#include "AbstractService.h"
#include "MoveService.h"
#include "FindService.h"

// set log & debug streams
std::ostream *logStream = &std::cout;
std::ostream *pdout = &std::cout;

// create service objects

//M4DStoreService storeService;

///////////////////////////////////////////////////////////////////////

M4DDcmProvider::M4DDcmProvider()
{
	findService = new M4DFindService();
	moveService = new M4DMoveService();
}

M4DDcmProvider::~M4DDcmProvider()
{
	delete ( (M4DFindService *) findService);
	delete ( (M4DMoveService *) moveService);
}

///////////////////////////////////////////////////////////////////////
// Theese function are only inline redirections to member functions
// of appropriate service objects

void
M4DDcmProvider::Find( 
		M4DDcmProvider::ResultSet &result,
		const string &patientName,
		const string &patientID,
		const string &modalities,
		const string &date) throw (...)
{
	static_cast<M4DFindService *>(findService)->FindForFilter( 
		result, patientName, patientID, modalities, date);
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::FindStudyInfo(
		const string &patientID,
		const string &studyID,
		StringVector &info) throw (...)
{
	static_cast<M4DFindService *>(findService)->FindStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::WholeFindStudyInfo(
		const string &patientID,
		const string &studyID,
		StudyInfo &info) throw (...)
{
	static_cast<M4DFindService *>(findService)->FindWholeStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::FindStudiesAboutPatient(  
		const string &patientID,
		ResultSet &result) throw (...)
{
	static_cast<M4DFindService *>(findService)->FindStudiesAboutPatient( 
		patientID, result);
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::GetImage(
		const string &patientID,
		const string &studyID,
		const string &serieID,
		const string &imageID,
		DicomObj &object) throw (...)
{
	static_cast<M4DMoveService *>(moveService)->MoveImage( 
		patientID, studyID, serieID, imageID, object);
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::GetImageSet(
		const string &patientID,
		const string &studyID,
		const string &serieID,
		DicomObjSet &result) throw (...)
{
	static_cast<M4DMoveService *>(moveService)->MoveImageSet( 
		patientID, studyID, serieID, result);
}

///////////////////////////////////////////////////////////////////////