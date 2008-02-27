#ifndef M4D_DICOM_OBJECT
#define M4D_DICOM_OBJECT

class M4DDicomObj
{
private:
	void *m_dataset;

public:
	// load & save
	void Load( string path) throw (...);
	void Save( string path)	throw (...);

	////////////////////////////////////////////////////////////
	// simple methods that returns values of each tag
	// patient level
	string GetPatientName( void) throw (...);
	bool	GetPatientSex( void) throw (...);
	string GetPatientBirthDate( void) throw (...);
	string GetPatientId( void) throw (...);

	// study level
	string GetStudyInstanceUID( void) throw (...);

	// series level
	string GetSeriesInstanceUID ( void) throw (...);
	string GetModality( void) throw (...);

	// image level
	void * GetPixelData( void) throw (...);
	// ...
	
	////////////////////////////////////////////////////////////
};

#endif