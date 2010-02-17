#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "GUI/widgets/m4dGUIMainWindow.h"

#include "Imaging/PipelineContainer.h"
#include "Imaging/ImageFactory.h"

#include "RGBGradientSliceViewerTexturePreparer.h"
#include "TypeDeclarations.h"
#include "SettingsBox.h"


class Notifier: public QObject, public M4D::Imaging::MessageReceiverInterface
{
	Q_OBJECT

  public:

	  Notifier ( QWidget *owner )
      : owner( owner ) 
    {}

	  void ReceiveMessage ( M4D::Imaging::PipelineMessage::Ptr msg, 
		                      M4D::Imaging::PipelineMessage::MessageSendStyle, 
		                      M4D::Imaging::FlowDirection )
	  {
		  if ( msg->msgID == M4D::Imaging::PMI_FILTER_UPDATED ) {
			  emit Notification();
		  }
	  }

  signals:
    
    void Notification ();

  protected:

	  QWidget	*owner;
};


class mainWindow: public M4D::GUI::m4dGUIMainWindow
{
	Q_OBJECT

  public:

    typedef M4D::Imaging::PipelineContainer PipelineType;
    typedef M4D::Imaging::APipeFilter FilterType;
    typedef M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > ConnectionType;
    typedef std::vector< M4D::Imaging::ConnectionInterface * > MultipleConnectionType;
    
    typedef M4D::Viewer::RGBGradientSliceViewerTexturePreparer< ElementType > RGBGradientTexturePreparer;

    static const unsigned SOURCE_NUMBER = 4;

	  mainWindow ();

  protected slots:

    void setSelectedViewerToSimple ();

    void setSelectedViewerToRGB ();

    void sourceSelected ();

  protected:

    void createPipeline ();
	  
    void process ( M4D::Imaging::ADataset::Ptr inputDataSet );

	  SettingsBox	*settings;
    
    Notifier *notifier;

	  PipelineType pipeline;

    FilterType *convertor, *registration, *segmentation, *analysis;

	  ConnectionType *inConnection, *registrationSegmentationConnection, *segmentationAnalysisConnection;
    MultipleConnectionType outConnection;

    RGBGradientTexturePreparer texturePreparer;
};


#endif // MAIN_WINDOW_H


