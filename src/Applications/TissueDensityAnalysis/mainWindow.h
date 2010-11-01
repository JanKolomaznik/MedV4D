#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/widgets/m4dGUIMainWindow.h"
#include "Imaging/filters/ImageConvertor.h"
#include "TDASphereSelection.h"
#include "SettingsBox.h"

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "BoneSegmentation"

//typedef int16	ElementType;
//const unsigned Dim = 3;
typedef M4D::Imaging::Image< ElementType, Dim > ImageType;
typedef M4D::Imaging::ThresholdingMaskFilter< ImageType > Thresholding;
typedef M4D::Imaging::MaskMedianFilter2D< Dim > Median2D;
//typedef M4D::Imaging::MaskMyFilter2D< Dim > MyFilter2D;
typedef M4D::Imaging::TDASphereSelection< ImageType > SphereSelectionFilter;
typedef M4D::Imaging::MaskSelection< ImageType > MaskSelectionFilter;
typedef M4D::Imaging::ConnectionTyped< ImageType > InConnection;
typedef M4D::Imaging::ImageConvertor< ImageType > InImageConvertor;


class Notifier : public QObject, public M4D::Imaging::MessageReceiverInterface
{
	Q_OBJECT
public:
	Notifier( QWidget *owner ): _owner( owner ) {}
	void ReceiveMessage(M4D::Imaging::PipelineMessage::Ptr 			        msg, 
		                  M4D::Imaging::PipelineMessage::MessageSendStyle /*sendStyle*/, 
		                  M4D::Imaging::FlowDirection				              /*direction*/
		)
	{
		if( msg->msgID == M4D::Imaging::PMI_FILTER_UPDATED ) {
			emit Notification();
		}
	}
signals:
	void Notification();
protected:
	QWidget	*_owner;
};

class mainWindow: public M4D::GUI::m4dGUIMainWindow
{
	Q_OBJECT

public:
 
	mainWindow ();
	void build();
	void switchToDefaultViewerDesktop ();


protected:

	void process ( M4D::Imaging::ADataset::Ptr inputDataSet );
	
	// nahradim defaultny viewer vlastni viewrem MySliceViewerWidget
	void createDefaultViewerDesktop ();
	
	void CreatePipeline();
	
	SettingsBox	*_settings;
	Notifier * _notifier;

	M4D::Imaging::PipelineContainer			_pipeline;
	M4D::Imaging::TDASphereSelection< ImageType >	*_filter;
	M4D::Imaging::APipeFilter		*_convertor;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >	*_inConnection;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >	*_inMaskConnection;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >	*_tmpConnection;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >	*_outConnection;

private:
};


#endif // MAIN_WINDOW_H


