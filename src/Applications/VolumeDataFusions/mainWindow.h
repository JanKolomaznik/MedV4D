#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/widgets/m4dGUIMainWindow.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/filters/ImageRegistration.h"
#include "SettingsBox.h"

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "VolumeDataFusions"

typedef int16	ElementType;
const unsigned Dim = 3;
typedef M4D::Imaging::Image< ElementType, Dim > ImageType;
typedef M4D::Imaging::ConnectionTyped< ImageType > InConnection;
typedef M4D::Imaging::ImageRegistration< ElementType, Dim > InImageRegistration;

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

protected:
	void
	process( M4D::Imaging::AbstractDataSet::Ptr inputDataSet );

	void
	CreatePipeline();

	SettingsBox	*_settings;
	Notifier * _notifier;

	M4D::Imaging::PipelineContainer			_pipeline;
	M4D::Imaging::AbstractPipeFilter		*_register[ SLICEVIEWER_INPUT_NUMBER ];
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage >	*_inConnection[ SLICEVIEWER_INPUT_NUMBER ];
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage >	*_outConnection[ SLICEVIEWER_INPUT_NUMBER ];

private:

};


#endif // MAIN_WINDOW_H


