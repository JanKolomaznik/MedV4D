#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/widgets/m4dGUIMainWindow.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/ImageFactory.h"

#include "SettingsBox.h"

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "SimpleProjection"



class Notifier : public QObject, public M4D::Imaging::MessageReceiverInterface
{
	Q_OBJECT
public:
	Notifier( QWidget *owner ): _owner( owner ) {}
	void
	ReceiveMessage( 
		M4D::Imaging::PipelineMessage::Ptr 			msg, 
		M4D::Imaging::PipelineMessage::MessageSendStyle 	/*sendStyle*/, 
		M4D::Imaging::FlowDirection				/*direction*/
		)
	{
		if( msg->msgID == M4D::Imaging::PMI_FILTER_UPDATED ) {
			emit Notification();
		}
	}

signals:
	void
	Notification();
protected:
	QWidget	*_owner;
};

class mainWindow: public M4D::GUI::m4dGUIMainWindow
{
	Q_OBJECT

public:

	mainWindow ();
	
	LevelSetFilterProperties properties_;
	
	typedef M4D::Imaging::ImageConvertor< ImageType > InImageConvertor;

protected:
	void
	process( M4D::Dicom::DicomObjSetPtr dicomObjSet );

	void
	CreatePipeline();

	SettingsBox	*_settings;
	Notifier	*_notifier;

	M4D::Imaging::PipelineContainer			_pipeline;
	M4D::Imaging::AbstractPipeFilter		*_filter;
	M4D::Imaging::AbstractPipeFilter		*_convertor;
	M4D::Imaging::AbstractPipeFilter		*_decimator;
	M4D::Imaging::ConnectionInterfaceTyped<M4D::Imaging::AbstractImage>	*_inConnection;
	M4D::Imaging::ConnectionInterfaceTyped<M4D::Imaging::AbstractImage>	*_outConnection;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage >	*_tmpConnection;

private:

};


#endif // MAIN_WINDOW_H


