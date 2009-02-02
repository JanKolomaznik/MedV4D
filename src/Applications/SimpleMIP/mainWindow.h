#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/m4dGUIMainWindow.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/filters/SimpleProjection.h"
#include "Imaging/filters/ImageConvertor.h"
#include "SettingsBox.h"

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "SimpleProjection"

typedef int16	ElementType;
typedef M4D::Imaging::Image< ElementType, 3 > ImageType;
typedef M4D::Imaging::SimpleProjection< ImageType > SimpleProjectionFilter;
typedef M4D::Imaging::ImageConvertor< ImageType > InImageConvertor;

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

protected:
	void
	process( M4D::Dicom::DcmProvider::DicomObjSetPtr dicomObjSet );

	void
	CreatePipeline();

	SettingsBox	*_settings;
	Notifier	*_notifier;

	M4D::Imaging::PipelineContainer			_pipeline;
	M4D::Imaging::AbstractPipeFilter		*_filter;
	M4D::Imaging::AbstractPipeFilter		*_convertor;
	M4D::Imaging::ConnectionInterfaceTyped<M4D::Imaging::AbstractImage>	*_inConnection;
	M4D::Imaging::ConnectionInterfaceTyped<M4D::Imaging::AbstractImage>	*_outConnection;

private:

};


#endif // MAIN_WINDOW_H


