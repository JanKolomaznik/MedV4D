#ifndef M4DDICOM_STORE_SERVICE
#define M4DDICOM_STORE_SERVICE

/**
 *  Implements service that performs C-STORE operatoin to DICOM server. It's purpose is to generate unique IDs and send new data to server.
 *  Behavior in a nutshell: Client (SCU) generates unique ID for sent (new) data, establish assotiatoin with a server (SCP) and sends the data to server.
 *  For more informations see DICOM doc ([ver]_08.pdf chapter 9.1.1) and coresponding annexes).
 *  Generation of unique IDs is based on prefix and the rest is delegated to DCMTK functions. More informations about unique UDs generation see DICOM doc.
 */

using namespace M4D::Dicom;

namespace M4D
{
namespace DicomInternal {

	class StoreService : AbstractService
	{
    friend class DcmProvider;

		static uint16 seriesCounter;
		static uint16 imageCounter;
		static OFString seriesInstanceUID;
		static OFString seriesNumber;
		static OFString accessionNumber;

		static void AddPresentationContext(T_ASC_Parameters *params,
			T_ASC_PresentationContextID presentationContextId, 
			const OFString& abstractSyntax,
			const OFList<OFString>& transferSyntaxList,
			T_ASC_SC_ROLE proposedRole) ;

		static void AddStoragePresentationContexts(
			T_ASC_Parameters *params, OFList<OFString>& sopClasses) ;

		static void	AddPresentationContext(T_ASC_Parameters *params,
			T_ASC_PresentationContextID presentationContextId,
			const OFString& abstractSyntax,
			const OFString& transferSyntax,
			T_ASC_SC_ROLE proposedRole);

		static OFBool IsaListMember(OFList<OFString>& lst, OFString& s);

    // supports for generation of UIDs. Taken from DCMTK.
		static int SecondsSince1970( void);
    // ...
		static OFString IntToString(int i);
    // ...
		static OFString MakeUID(OFString basePrefix, int counter);
    // ...
		static OFBool UpdateStringAttributeValue(
			DcmItem* dataset, const DcmTagKey& key, OFString& value);
    // ...
		static void ReplaceSOPInstanceInformation(DcmDataset* dataset);
    // ...
    void CopyNeccessaryAttribs( DcmDataset *source, DcmDataset *dest);

    // supports
		static void	ProgressCallback(void * /*callbackData*/,
			T_DIMSE_StoreProgress *progress,
			T_DIMSE_C_StoreRQ * /*req*/);

		void StoreSCP(DcmDataset *data);
		
		void StoreObject( 
			const Dicom::DcmProvider::DicomObj &objectToCopyAttribsFrom,
			Dicom::DcmProvider::DicomObj &objectToStore);

    // ctor & dtor
		StoreService();
		~StoreService();
	};

}
}

#endif

