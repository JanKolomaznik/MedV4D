#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "MedV4D/GUI/widgets/m4dGUIMainWindow.h"

#include "Imaging/PipelineContainer.h"
#include "Imaging/ImageFactory.h"

#include "RGBGradientSliceViewerTexturePreparer.h"
#include "TypeDeclarations.h"
#include "SettingsBox.h"
#include "PlotBox.h"

/**
 * Notifier for indicating end of the pipeline computation.
 */
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


/**
 * The main window of the application.
 */
class MainWindow: public M4D::GUI::m4dGUIMainWindow
{
	Q_OBJECT

  public:

    typedef M4D::Imaging::PipelineContainer PipelineType;
    typedef M4D::Imaging::APipeFilter FilterType;
    typedef M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > ConnectionType;
    typedef std::vector< M4D::Imaging::ConnectionInterface * > MultipleConnectionType;
    
    typedef M4D::Viewer::RGBGradientSliceViewerTexturePreparer< ElementType > RGBGradientTexturePreparer;

    static const unsigned SOURCE_NUMBER = 4;

    /**
     * The main window constructor.
     */
	  MainWindow ();

  protected slots:

    /**
     * Slot for setting viewers connected to the output to simple mode: setting their 
     * texture preparer to simple.
     */
    void SetSelectedViewerToSimple ();

    /**
     * Slot for setting viewers connected to the output to RGB (color mapped) mode: setting their 
     * texture preparer to custom (attribute texturePreparer).
     */
    void SetSelectedViewerToRGB ();

    /**
     * Slot for setting viewers connected to the output to point picker mode (for TIC plot tool).
     * 
     * @param toolEnabled flag indicating whether the TIC plot tool is enabled
     */
    void SetSelectedViewerToPoint ( bool toolEnabled );

    /**
     * Slot for setting viewers connected to the output to region picker mode (for see-through tool).
     * 
     * @param toolEnabled flag indicating whether the see-through tool is enabled
     */
    void SetSelectedViewerToRegion ( bool toolEnabled );

    /**
     * Slot for modifying source selection behaviour.
     */
    void SourceSelected ();

  protected:

    /**
	   * Creates Pipeline - filters, connections.
	   */
    void CreatePipeline ();
	  
    /**
	   * Processes dataset.
     *
	   * @param inputDataSet smart pointer to the dataset to be processed
	   */
    void process ( M4D::Imaging::ADataset::Ptr inputDataSet );

	  /// Pointer to the Perfusion Studies settings widget.
    SettingsBox	*settings;

    /// Pointer to TIC plot tool's plotting widget.
    PlotBox	*plot;
    
    /// Pointer to notifier for indicating end of the pipeline computation.
    Notifier *notifier;

	  /// the pipeline.
    PipelineType pipeline;

    /// Pointers to the filters.
    FilterType *convertor, *registration, *segmentation, *analysis;

	  /// Connections between the filters.
    ConnectionType *inConnection, *registrationSegmentationConnection, *segmentationAnalysisConnection;
    MultipleConnectionType outConnection;

    /// The custom texture preparer - for Parameter Maps type of visualization.
    RGBGradientTexturePreparer texturePreparer;
};


#endif // MAIN_WINDOW_H


