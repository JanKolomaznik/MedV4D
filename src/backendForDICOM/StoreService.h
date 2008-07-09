#ifndef M4DDICOM_STORE_SERVICE
#define M4DDICOM_STORE_SERVICE

/**
 *  Implements service that performs C-STORE operatoin to DICOM server. It has generate unique IDs and send new data to server.
 *  For more informations see DICOM doc ([ver]_08.pdf chapter 9.1.1) and coresponding annexes).
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

		static int SecondsSince1970( void);

		static OFString IntToString(int i);

		static OFString MakeUID(OFString basePrefix, int counter);

		static OFBool UpdateStringAttributeValue(
			DcmItem* dataset, const DcmTagKey& key, OFString& value);

		static void ReplaceSOPInstanceInformation(DcmDataset* dataset);

		static void	ProgressCallback(void * /*callbackData*/,
			T_DIMSE_StoreProgress *progress,
			T_DIMSE_C_StoreRQ * /*req*/);

		void StoreSCP(DcmDataset *data);

		void CopyNeccessaryAttribs( DcmDataset *source, DcmDataset *dest);

		StoreService();
		~StoreService();
		
		void StoreObject( 
			const Dicom::DcmProvider::DicomObj &objectToCopyAttribsFrom,
			Dicom::DcmProvider::DicomObj &objectToStore);
	};

}
}

#endif

