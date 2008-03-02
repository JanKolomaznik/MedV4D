#ifndef M4D_DICOM_OBJECT
#define M4D_DICOM_OBJECT

class M4DDicomObj
{
private:
	//friend class M4DMoveService;
	
public:
	// TODO - make private
	void *m_dataset;
	bool m_loaded;

	inline bool IsLoaded( void) { return m_loaded; }

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

	M4DDicomObj();
};

#endif