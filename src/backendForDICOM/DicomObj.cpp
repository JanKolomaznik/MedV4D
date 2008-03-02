
#include <string>

using namespace std;

#include "M4DDICOMServiceProvider.h"

#include "dcmtk/dcmdata/dcdeftag.h"
#include "dcmtk/dcmdata/dcfilefo.h"

#include "main.h"

///////////////////////////////////////////////////////////////////////

M4DDcmProvider::DicomObj::DicomObj()
{
	m_dataset = NULL;
	m_loaded = false;
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::DicomObj::Save( const string &path)	
	throw (...)
{
	if( m_dataset == NULL)
		return;

	//DcmFileFormat file;
	DcmDataset *dataSet = static_cast<DcmDataset *>(m_dataset);

	//E_EncodingType    opt_sequenceType = EET_ExplicitLength;
	//E_PaddingEncoding opt_paddingType = EPD_withoutPadding;
	//E_GrpLenEncoding  opt_groupLength = EGL_recalcGL;
	//unsigned int opt_itempad = 0;
	//unsigned int opt_filepad = 0;
	//OFBool            opt_useMetaheader = OFTrue;

	OFCondition cond = dataSet->saveFile( path.c_str());
	if (cond.bad())
	{
		LOG( "Cannot write file:" << path);
		throw new bad_exception();
	}
}

///////////////////////////////////////////////////////////////////////

void
M4DDcmProvider::DicomObj::Load( const string &path) throw (...)
{
	DcmDataset *dataSet = new DcmDataset();
	OFCondition cond = dataSet->loadFile( path.c_str());

	if (cond.bad())
	{
		LOG( "Cannot load the file: " << path);
		throw new bad_exception();
	}

	m_dataset = (void *)dataSet;
}

///////////////////////////////////////////////////////////////////////

string
M4DDcmProvider::DicomObj::GetPatientName( void) throw (...)
{
	OFString s;
	static_cast<DcmDataset *>(m_dataset)->findAndGetOFString( 
		DCM_PatientsName, s );

	return string(s.c_str());
}

///////////////////////////////////////////////////////////////////////