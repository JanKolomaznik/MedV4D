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
DcmProvider::LocalFind( 
			DcmProvider::ResultSet &result,
      const std::string &path)
{
  static_cast<LocalService *>(m_localService)->Find( result, path);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::LocalFindStudyInfo( 
      const std::string &patientID,
			const std::string &studyID,
      SerieInfoVector &info)
{
  static_cast<LocalService *>(m_localService)->FindStudyInfo( 
    info, patientID, studyID);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::LocalGetImageSet(
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
		SerieInfoVector &info) 
{
	static_cast<FindService *>(m_findService)->FindStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindStudyAndImageInfo(
		const string &patientID,
		const string &studyID,
		StudyInfo &info) 
{
	static_cast<FindService *>(m_findService)->FindWholeStudyInfo(
		patientID, studyID, info);
}

///////////////////////////////////////////////////////////////////////

void
DcmProvider::FindAllPatientStudies(  
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
		DicomObjSet &result,
    DicomObj::ImageLoadedCallback on_loaded) 
{
	static_cast<MoveService *>(m_moveService)->MoveImageSet( 
		patientID, studyID, serieID, result, on_loaded);
}

///////////////////////////////////////////////////////////////////////

} // namespace
}
