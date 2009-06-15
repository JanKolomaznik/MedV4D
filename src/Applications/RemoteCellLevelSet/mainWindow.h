#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/widgets/m4dGUIMainWindow.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/filters/ImageConvertor.h"


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
	
	typedef M4D::Imaging::Image< uint16, 3 > VeiwImageType;
	typedef M4D::Imaging::ImageConvertor< VeiwImageType > ViewImageConvertor;

protected:
	void
	process(M4D::Imaging::AbstractDataSet::Ptr inputDataSet );

	void
	CreatePipeline();

	SettingsBox	*_settings;
	Notifier	*_notifier;
	Notifier	*_adapterDoneNotifier;

	M4D::Imaging::PipelineContainer			_pipeline;
	M4D::Imaging::AbstractPipeFilter		*_filter;
	M4D::Imaging::AbstractPipeFilter		*_convertor;
	M4D::Imaging::AbstractPipeFilter		*_decimatedImConvertor;
	M4D::Imaging::AbstractPipeFilter		*_resultImConvertor;
	M4D::Imaging::AbstractPipeFilter		*_decimator;
	M4D::Imaging::ConnectionInterfaceTyped<M4D::Imaging::AbstractImage>	*_inConnection;
	M4D::Imaging::ConnectionInterfaceTyped<M4D::Imaging::AbstractImage>	*_outConnection;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage >	*_tmpConnection;
	
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage >	*_decim2castConnection;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage >	*_castOutConnection;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage >	*_remote2castConnection;
	
protected slots:
	void OnAdapterDone();
	
private:

};


#endif // MAIN_WINDOW_H


