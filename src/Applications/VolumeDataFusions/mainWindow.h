#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/widgets/m4dGUIMainWindow.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/filters/ImageRegistration.h"


#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "VolumeDataFusions"

typedef int16	ElementType;
const unsigned Dim = 3;
typedef M4D::Imaging::Image< ElementType, Dim > ImageType;
typedef M4D::Imaging::ConnectionTyped< ImageType > InConnection;
typedef M4D::Imaging::ImageRegistration< ElementType, Dim > InImageRegistration;

class SettingsBox;

#include "SettingsBox.h"

class Notifier : public QObject, public M4D::Imaging::MessageReceiverInterface
{
	Q_OBJECT
public:
	Notifier( unsigned number, QWidget *owner ): _number( number ), _owner( owner ) {}
	void ReceiveMessage(M4D::Imaging::PipelineMessage::Ptr 			        msg, 
		                  M4D::Imaging::PipelineMessage::MessageSendStyle /*sendStyle*/, 
		                  M4D::Imaging::FlowDirection				              /*direction*/
		)
	{
		if( msg->msgID == M4D::Imaging::PMI_FILTER_UPDATED ) {
			emit Notification( _number );
		}
	}
signals:
	void Notification( unsigned );
protected:
	unsigned _number;
	QWidget	*_owner;
};

/**
 * MedV4D GUI main window
 */
class mainWindow: public M4D::GUI::m4dGUIMainWindow
{
	Q_OBJECT

public:

	mainWindow ();

	/**
	 * Plug the output connection to an input connection of a viewer
	 *  @param inputNumber the number of the input image (respectively the number of the output connection) 
	 *  @param portNumber the viewer's input port number
	 */
	void
	OutConnectionToViewerPort( uint32 inputNumber, uint32 portNumber );

public slots:

	/**
	 * Clear selected dataset
	 */
	void
	ClearDataset();

protected:

	/**
	 * Process dataset
	 *  @param inputDataSet smart pointer to the dataset to be processed
	 */
	void
	process( M4D::Imaging::ADataset::Ptr inputDataSet );

	/**
	 * Create Pipeline
	 */
	void
	CreatePipeline();

	// settings box
	SettingsBox								*_settings;

	// notifiers about compatition
	Notifier								*_notifier[ SLICEVIEWER_INPUT_NUMBER ];

	// pipeline
	M4D::Imaging::PipelineContainer								_pipeline;

	// registration filters
	M4D::Imaging::APipeFilter								*_register[ SLICEVIEWER_INPUT_NUMBER ];

	// connections
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >	*_inConnection[ SLICEVIEWER_INPUT_NUMBER ];
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage >	*_outConnection[ SLICEVIEWER_INPUT_NUMBER ];

private:

};


#endif // MAIN_WINDOW_H


