#ifndef M4DDICOM_STORE_SERVICE
#define M4DDICOM_STORE_SERVICE

/**
 *  Implements C-FIND service to DICOM server
 */

using namespace std;

class M4DStoreService : M4DAbstractService
{
private :

	static uint16 seriesCounter;
	static uint16 imageCounter;
	static OFString seriesInstanceUID;
	static OFString seriesNumber;
	static OFString accessionNumber;

	static void AddPresentationContext(T_ASC_Parameters *params,
		T_ASC_PresentationContextID presentationContextId, 
		const OFString& abstractSyntax,
		const OFList<OFString>& transferSyntaxList,
		T_ASC_SC_ROLE proposedRole) throw(...);

	static void AddStoragePresentationContexts(
		T_ASC_Parameters *params, OFList<OFString>& sopClasses) throw(...);

	static void	AddPresentationContext(T_ASC_Parameters *params,
		T_ASC_PresentationContextID presentationContextId,
		const OFString& abstractSyntax,
		const OFString& transferSyntax,
		T_ASC_SC_ROLE proposedRole) throw(...);

	static OFBool IsaListMember(OFList<OFString>& lst, OFString& s);

	static int SecondsSince1970( void);

	static OFString IntToString(int i);

	static OFString MakeUID(OFString basePrefix, int counter);

	static OFBool UpdateStringAttributeValue(
		DcmItem* dataset, const DcmTagKey& key, OFString& value);

	static void ReplaceSOPInstanceInformation(DcmDataset* dataset);

	static void	ProgressCallback(void * /*callbackData*/,
		T_DIMSE_StoreProgress *progress,
		T_DIMSE_C_StoreRQ * /*req*/);

	void StoreSCP(DcmDataset *data) throw(...);

	void CopyNeccessaryAttribs( DcmDataset *source, DcmDataset *dest);

public:
	M4DStoreService();
	~M4DStoreService();
	
	void StoreObject( 
		const M4DDcmProvider::DicomObj &objectToCopyAttribsFrom,
		M4DDcmProvider::DicomObj &objectToStore) throw(...);
};

#endif