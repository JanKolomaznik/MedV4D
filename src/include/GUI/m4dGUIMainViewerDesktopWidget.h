#ifndef M4D_GUI_MAIN_VIEWER_DESKTOP_H
#define M4D_GUI_MAIN_VIEWER_DESKTOP_H

#include <QtGui>

#include "GUI/m4dGUIAbstractViewerWidget.h"
#include "GUI/m4dGUISliceViewerWidget.h"
#include "GUI/m4dGUIVtkViewerWidget.h"


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

    /** 
     * Main Viewer Desktop constructor.
     *
     * @param parent pointer to the parent widget - default is 0
     */
    m4dGUIMainViewerDesktopWidget ( QWidget *parent = 0 );

    /** 
     * Changes the Desktop's layout.
     *
     * @param rows number of rows in the new layout
     * @param columns number of columns in the new layout
     */
    void setDesktopLayout( const int rows, const int columns );

    M4D::Viewer::m4dGUIAbstractViewerWidget *getSelectedViewer() { return selectedViewer; }
    M4D::Viewer::m4dGUIAbstractViewerWidget *getPrevSelectedViewer() { return prevSelectedViewer; }

    M4D::Viewer::m4dGUIVtkViewerWidget *getVtkRenderWindowWidget() { return vtkRenderWindowWidget; }

  private slots:

    void selectedChanged ( unsigned index );

  signals:

    void propagateFeatures ();

  private:

    M4D::Viewer::m4dGUIVtkViewerWidget *vtkRenderWindowWidget;
    M4D::Viewer::m4dGUISliceViewerWidget *glWidget;

    M4D::Imaging::Image< uint32, 3 >::Ptr inputImage;
    M4D::Imaging::ImageConnectionSimple< M4D::Imaging::Image< uint32, 3 > > prodconn;

    std::vector< M4D::Viewer::m4dGUIAbstractViewerWidget * > viewers;
    M4D::Viewer::m4dGUIAbstractViewerWidget *selectedViewer;
    M4D::Viewer::m4dGUIAbstractViewerWidget *prevSelectedViewer;
};

#endif // M4D_GUI_MAIN_VIEWER_DESKTOP_H

