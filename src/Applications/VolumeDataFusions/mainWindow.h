#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/widgets/m4dGUIMainWindow.h"
#include "Imaging/PipelineContainer.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/filters/ThresholdingFilter.h"
#include "Imaging/filters/MaskMedianFilter.h"
#include "Imaging/filters/MaskSelection.h"
#include "Imaging/filters/ImageRegistration.h"
#include "SettingsBox.h"

#define ORGANIZATION_NAME     "MFF"
#define APPLICATION_NAME      "VolumeDataFusions"

#define FUSION_IMAGE_NUMBER			12

typedef int16	ElementType;
const unsigned Dim = 3;
typedef M4D::Imaging::Image< ElementType, Dim > ImageType;
typedef M4D::Imaging::ThresholdingMaskFilter< ImageType > Thresholding;
typedef M4D::Imaging::MaskMedianFilter2D< Dim > Median2D;
typedef M4D::Imaging::MaskSelection< ImageType > MaskSelectionFilter;
typedef M4D::Imaging::ConnectionTyped< ImageType > InConnection;
typedef M4D::Imaging::ImageTransform< ElementType, Dim > InImageTransform;
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
	M4D::Imaging::AbstractPipeFilter		*_filter;
	M4D::Imaging::AbstractPipeFilter		*_convertor;
	M4D::Imaging::AbstractPipeFilter		*_transformer;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage >	*_inConnection;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage >	*_transConnection;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage >	*_tmpConnection;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AbstractImage >	*_outConnection;

private:

};


#endif // MAIN_WINDOW_H


