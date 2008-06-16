#include <queue>

#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmnet/dimse.h>
#include <dcmtk/dcmnet/diutil.h>
#include <dcmtk/dcmdata/dcdeftag.h>

#include "boost/filesystem/path.hpp"

#include "Common.h"

#include "DicomAssoc.h"
#include "dicomConn/DICOMServiceProvider.h"
#include "AbstractService.h"
#include "MoveService.h"
#include "FindService.h"
#include "LocalService.h"

namespace M4D
{
using namespace DicomInternal;


// create service objects

//StoreService storeService;

namespace Dicom {

///////////////////////////////////////////////////////////////////////

DcmProvider::DcmProvider()
{
	m_findService = new FindService();
	m_moveService = new MoveService();
  m_localService = new LocalService();
}

DcmProvider::~DcmProvider()
{
	delete ( (FindService *) m_findService);
	delete ( (MoveService *) m_moveService);
  delete ( (LocalService *) m_localService);
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
	static_cast<FindService *>(m_findService)->FindForFilter( 
		result, patientForeName, patientSureName, 
    patientID, modalities, dateFrom, dateTo);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindInFolder( 
			DcmProvider::ResultSet &result,
      const std::string &path)
{
  static_cast<LocalService *>(m_localService)->Find( result, path);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::GetLocalImageSet(
      const std::string &patientID,
			const std::string &studyID,
			const std::string &serieID,
			DicomObjSet &result)
{
  static_cast<LocalService *>(m_localService)->GetImageSet( 
    patientID, studyID, serieID, result);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindStudyInfo(
		const string &patientID,
		const string &studyID,
		StringVector &info) 
{
	static_cast<FindService *>(m_findService)->FindStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::WholeFindStudyInfo(
		const string &patientID,
		const string &studyID,
		StudyInfo &info) 
{
	static_cast<FindService *>(m_findService)->FindWholeStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindStudiesAboutPatient(  
		const string &patientID,
		ResultSet &result) 
{
	static_cast<FindService *>(m_findService)->FindStudiesAboutPatient( 
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
	static_cast<MoveService *>(m_moveService)->MoveImage( 
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
	static_cast<MoveService *>(m_moveService)->MoveImageSet( 
		patientID, studyID, serieID, result);
}

///////////////////////////////////////////////////////////////////////

} // namespace
}
