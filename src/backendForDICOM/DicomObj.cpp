
#include <string>

using namespace std;

#include "M4DDICOMServiceProvider.h"

#include "dcmtk/dcmdata/dcdeftag.h"
#include "dcmtk/dcmdata/dcfilefo.h"

M4DDcmProvider::DicomObj::DicomObj()
{
	m_dataset = NULL;
	m_loaded = false;
}

void
M4DDcmProvider::DicomObj::Save( string path)	
	throw (...)
{
	if( m_dataset == NULL)
		return;

	//DcmFileFormat file;
	DcmDataset *dataSet = static_cast<DcmDataset *>(m_dataset);

	/*E_EncodingType    opt_sequenceType = EET_ExplicitLength;
	E_PaddingEncoding opt_paddingType = EPD_withoutPadding;
	E_GrpLenEncoding  opt_groupLength = EGL_recalcGL;
	unsigned int opt_itempad = 0;
	unsigned int opt_filepad = 0;
	OFBool            opt_useMetaheader = OFTrue;*/

	/*OFCondition cond = file.saveFile(
		path.c_str(), dataSet->getOriginalXfer(),
		opt_sequenceType, opt_groupLength,
		opt_paddingType, (Uint32)opt_filepad, (Uint32)opt_itempad, !opt_useMetaheader);*/

	OFCondition cond = dataSet->saveFile( path.c_str());
	if (cond.bad())
		throw new bad_exception( "Cannot write image file");
}