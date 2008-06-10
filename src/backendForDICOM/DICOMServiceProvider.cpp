#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmnet/dimse.h>
#include <dcmtk/dcmnet/diutil.h>
#include <dcmtk/dcmdata/dcdeftag.h>

#include "Common.h"

#include "DicomAssoc.h"
#include "dicomConn/DICOMServiceProvider.h"
#include "AbstractService.h"
#include "MoveService.h"
#include "FindService.h"

namespace M4D
{
using namespace DicomInternal;


// create service objects

//StoreService storeService;

namespace Dicom {

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
		const string &patientForeName,
    const string &patientSureName,
		const string &patientID,
		const StringVector &modalities,
		const string &dateFrom,
		const string &dateTo) 
{
	static_cast<FindService *>(findService)->FindForFilter( 
		result, patientForeName, patientSureName, 
    patientID, modalities, dateFrom, dateTo);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindStudyInfo(
		const string &patientID,
		const string &studyID,
		StringVector &info) 
{
	static_cast<FindService *>(findService)->FindStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::WholeFindStudyInfo(
		const string &patientID,
		const string &studyID,
		StudyInfo &info) 
{
	static_cast<FindService *>(findService)->FindWholeStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindStudiesAboutPatient(  
		const string &patientID,
		ResultSet &result) 
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
		DicomObj &object) 
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
		DicomObjSet &result) 
{
	static_cast<MoveService *>(moveService)->MoveImageSet( 
		patientID, studyID, serieID, result);
}

///////////////////////////////////////////////////////////////////////

} // namespace
}
