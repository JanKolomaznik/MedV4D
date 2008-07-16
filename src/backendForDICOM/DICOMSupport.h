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
  // patient info
  ds->findAndGetOFString( DCM_PatientID, str);
	row->patientID = str.c_str();

	ds->findAndGetOFString( DCM_PatientsName, str);
	row->name = str.c_str();

	ds->findAndGetOFString( DCM_PatientsBirthDate, str);
	row->birthDate = str.c_str();

	ds->findAndGetOFString( DCM_PatientsSex, str);
	row->sex = (str == "M");	// M = true

	// study info
	ds->findAndGetOFString( DCM_StudyInstanceUID, str);
	row->studyID = str.c_str();

	ds->findAndGetOFString( DCM_StudyDate, str);
	row->date = str.c_str();

  ds->findAndGetOFString( DCM_StudyTime, str);
  row->time = str.c_str();

	ds->findAndGetOFString( DCM_Modality, str);
	row->modality = str.c_str();

  ds->findAndGetOFString( DCM_StudyDescription, str);
	row->description = str.c_str();

  ds->findAndGetOFString( DCM_ReferringPhysiciansName, str);
  row->referringMD = str.c_str();
}

/**
 * Retrive data for DcmProvider::SerieInfo from given dataSet
 */
static void GetSeriesInfo( DcmDataset *ds, DcmProvider::SerieInfo *sInfo)
{
  OFString str;
	// Parse the response
	ds->findAndGetOFString( DCM_SeriesInstanceUID, str);
  sInfo->id = str.c_str();

  ds->findAndGetOFString( DCM_SeriesDescription, str);
  sInfo->description = str.c_str();
}
	
#endif

