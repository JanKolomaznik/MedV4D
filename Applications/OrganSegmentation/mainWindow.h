#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "MedV4D/GUI/widgets/m4dGUIMainWindow.h"
#include "SegmentationWidget.h"
#include "SettingsBox.h"
#include "SegmentationTypes.h"
#include "SegmentationManager.h"

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "OrganSegmentation"

/*typedef int16	ElementType;
const unsigned Dim = 3;
typedef M4D::Imaging::Image< ElementType, Dim > ImageType;
typedef M4D::Imaging::ThresholdingMaskFilter< ImageType > Thresholding;
typedef M4D::Imaging::MaskMedianFilter2D< Dim > Median2D;
typedef M4D::Imaging::MaskSelection< ImageType > MaskSelectionFilter;
typedef M4D::Imaging::ImageConnection< ImageType > InConnection;
typedef M4D::Imaging::ImageConvertor< ImageType > InImageConvertor;
*/

//class Notifier : public QObject, public M4D::Imaging::MessageReceiverInterface
//{
//	Q_OBJECT
//public:
//	Notifier( QWidget *owner ): _owner( owner ) {}
//	void
//	ReceiveMessage( 
//		M4D::Imaging::PipelineMessage::Ptr 			msg, 
//		M4D::Imaging::PipelineMessage::MessageSendStyle 	/*sendStyle*/, 
//		M4D::Imaging::FlowDirection				/*direction*/
//		)
//	{
//		if( msg->msgID == M4D::Imaging::PMI_FILTER_UPDATED ) {
//			emit Notification();
//		}
//	}
//
//signals:
//	void
//	Notification();
//protected:
//	QWidget	*_owner;
//};

class mainWindow: public M4D::GUI::m4dGUIMainWindow
{
	Q_OBJECT

public:

	mainWindow ();

public slots:
	/*void
	SetSegmentationSlot( uint32 segType );*/

	void
	SetToDefaultSlot();

	void
	ManagerActivatedSlot( ManagerActivationInfo info );

	void
	ProcessResults( ResultsInfo info );

	void
	ErrorMessageSlot( const QString &message )
	{
		QMessageBox::critical( this, "Problem...", message );
	}
protected:
	void
	process ( M4D::Imaging::ADataset::Ptr inputDataset );

	SegmentationViewerWidget	*_segmentationViewerWidget;
	SettingsBox	*_settings;
	//Notifier	*_notifier;
private:

};



#endif // MAIN_WINDOW_H


