#ifndef M4D_DICOM_SUPPORT
#define M4D_DICOM_SUPPORT

// NOTE: dicomConn/DICOMServiceProvider.h has to be already included when used
// as well as DCMToolkit dataset headers

/**
 * Retrive data that are displayed in searching table from dataSet. 
 * Called from search services to fill table row.
 */
static void GetTableRowFromDataSet( 
   DcmDataset *ds, DcmProvider::TableRow *row)
{
  OFString str;
	ds->findAndGetOFString( DCM_PatientsName, str);
	row->patientName = str.c_str();

	ds->findAndGetOFString( DCM_PatientID, str);
	row->patentID = str.c_str();

	ds->findAndGetOFString( DCM_PatientsBirthDate, str);
	row->patientBirthDate = str.c_str();

	ds->findAndGetOFString( DCM_PatientsSex, str);
	row->patientSex = (str == "M");	// M = true

	// study info
	ds->findAndGetOFString( DCM_StudyInstanceUID, str);
	row->studyID = str.c_str();

	ds->findAndGetOFString( DCM_StudyDate, str);
	row->studyDate = str.c_str();

	ds->findAndGetOFString( DCM_Modality, str);
	row->modality = str.c_str();
}
	
#endif

