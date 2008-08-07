#ifndef M4D_GUI_MAIN_VIEWER_DESKTOP_H
#define M4D_GUI_MAIN_VIEWER_DESKTOP_H

#include <QtGui>

#include "GUI/m4dGUIAbstractViewerWidget.h"
#include "GUI/m4dGUISliceViewerWidget.h"
#include "GUI/m4dGUIVtkViewerWidget.h"


namespace M4D {
namespace GUI {

/**
 * @class m4dGUIMainViewerDesktopWidget m4dGUIMainViewerDesktopWidget.h
 * Class representing the Main Viewer Desktop - containing abstract viewers.
 * It can manage its own layout, with selection of specific viewer - it causes
 * changes in toolBars and controls depending on the type of the viewer.
 */
class m4dGUIMainViewerDesktopWidget: public QWidget
{
  Q_OBJECT

  public:

    typedef enum { SLICE_VIEWER, VTK_VIEWER } ViewerType;

    struct Viewer {
      ViewerType type;
      M4D::Viewer::m4dGUIAbstractViewerWidget *viewerWidget;
      /// checked tool (index) for given viewer - for left mouse button
      unsigned checkedLeftButtonTool;
      /// checked tool (index) for given viewer - for right mouse button
      unsigned checkedRightButtonTool;
    };

    /** 
     * Main Viewer Desktop constructor.
     *
     * @param parent pointer to the parent widget - default is 0
     */
    m4dGUIMainViewerDesktopWidget ( QWidget *parent = 0 );

    ViewerType getSelectedViewerType () const { return selectedViewer->type; }
    M4D::Viewer::m4dGUIAbstractViewerWidget *getSelectedViewerWidget () const { return selectedViewer->viewerWidget; }
    unsigned getSelectedViewerLeftTool () const { return selectedViewer->checkedLeftButtonTool; }
    unsigned getSelectedViewerRightTool () const { return selectedViewer->checkedRightButtonTool; }
    
    void replaceSelectedViewerWidget ( ViewerType type, M4D::Viewer::m4dGUIAbstractViewerWidget *replacedViewer );
    void setSelectedViewerLeftTool ( unsigned value ) { selectedViewer->checkedLeftButtonTool = value; }
    void setSelectedViewerRightTool ( unsigned value ) { selectedViewer->checkedRightButtonTool = value; }

    M4D::Viewer::m4dGUIAbstractViewerWidget *getPrevSelectedViewerWidget () const { return prevSelectedViewer->viewerWidget; }

    void addSource ( M4D::Imaging::ConnectionInterface *conn, const char *pipelineDescription,
                     const char *connectionDescription );

  private slots:

    /** 
     * Slot for changing the Desktop's layout.
     *
     * @param rows number of rows in the new layout
     * @param columns number of columns in the new layout
     */
    void setDesktopLayout ( const unsigned rows, const unsigned columns );

    void selectedChanged ( unsigned index );

    /** 
     * Slot for Source comboBox activity - should be connected to it's activated( int ) signal - it's
     * setting input port for selected viewer widget.
     *
     * @param index index of selected source (in comboBox and also in sources vector)
     */
    void sourceSelected ( int index );

  signals:

    void propagateFeatures ();
    void sourceAdded ( const QString &pipelineDescription, const QString &connectionDescription );

  private:

    M4D::Imaging::Image< uint32, 3 >::Ptr inputImage;
    M4D::Imaging::ImageConnection< M4D::Imaging::Image< uint32, 3 > > prodconn;

    std::vector< Viewer * > viewers;
    Viewer *selectedViewer;
    Viewer *prevSelectedViewer;

    /**
     * Vector of registered sources - possible connections, where can be plugged a viewer. Can be selected
     * through comboBox in toolBar. 
     */ 
    std::vector< M4D::Imaging::ConnectionInterface * > sources;

    unsigned layoutRows, layoutColumns;
};

} // namespace GUI
} // namespace M4D

#endif // M4D_GUI_MAIN_VIEWER_DESKTOP_H

