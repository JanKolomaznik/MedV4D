/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file m4dGUIMainViewerDesktopWidget.h 
 * @{ 
 **/

#ifndef M4D_GUI_MAIN_VIEWER_DESKTOP_H
#define M4D_GUI_MAIN_VIEWER_DESKTOP_H

#include <QtGui>

#include "utils/ViewerFactory.h"


namespace M4D {
namespace GUI {

/**
 * Class representing the Main Viewer Desktop - containing abstract viewers.
 * It can manage its own layout, with selection of specific viewer - it causes
 * changes in toolBars and controls depending on the type of the viewer.
 */
class m4dGUIMainViewerDesktopWidget: public QWidget
{
  Q_OBJECT

  public:

    /**
     * Structure representing a viewer (it contains the viewer widget, which 
     * tools were checked for it, index of its source - pipeline connection).
     */
    struct Viewer {
      /// Viewer widget.
      M4D::Viewer::m4dGUIAbstractViewerWidget *viewerWidget;
      /// Checked tool (index) for given viewer - for left mouse button.
      unsigned checkedLeftButtonTool;
      /// Checked tool (index) for given viewer - for right mouse button.
      unsigned checkedRightButtonTool;
      /// Index of selected source (in comboBox and also in sources vector).
      int sourceIdx;
    };

    /**
     * Structure representing registered sources - possible connections, where can be plugged a viewer -
     * + event handlers, which should be active on the viewer connected to particular source.
     */
    struct Source {
      /// Ctor.
      Source ( Imaging::ConnectionInterface *conn, M4D::Viewer::m4dGUIViewerEventHandlerInterface *hnd )
        : conn( conn ), hnd( hnd )
      {}
      /// Pointer to the connection.
      Imaging::ConnectionInterface *conn;
      /// Pointer to the viewer event handler.
      M4D::Viewer::m4dGUIViewerEventHandlerInterface *hnd;
    };


    /** 
     * Main Viewer Desktop constructor.
     *
     * @param rows number of rows in the layout of desktop widget
     * @param columns number of columns in the layout of desktop widget
     * @param viewerFactory pointer to the Viewer Factory - which will create viewers to the desktop
     * (will be deleted in the Main Viewer Desktop destructor)
     * @param parent pointer to the parent widget - default is 0
     */
    m4dGUIMainViewerDesktopWidget ( const unsigned rows, const unsigned columns, 
                                    ViewerFactory *viewerFactory, QWidget *parent = 0 );

    /** 
     * Main Viewer Desktop destructor.
     */
    ~m4dGUIMainViewerDesktopWidget ();

    /** 
     * Getter to the selected viewer's viewer widget.
     *
     * @return pointer to the viewer widget of the selected viewer
     */
    M4D::Viewer::m4dGUIAbstractViewerWidget *getSelectedViewerWidget () const { return selectedViewer->viewerWidget; }

    /** 
     * Getter to the selected viewer's checked tool (index) - for left mouse button.
     *
     * @return index of the checked tool (for left mouse button) of the selected viewer
     */
    unsigned getSelectedViewerLeftTool () const { return selectedViewer->checkedLeftButtonTool; }

    /** 
     * Getter to the selected viewer's checked tool (index) - for right mouse button.
     *
     * @return index of the checked tool (for right mouse button) of the selected viewer
     */
    unsigned getSelectedViewerRightTool () const { return selectedViewer->checkedRightButtonTool; }

    /** 
     * Getter to the selected viewer's index of the selected source (in comboBox and also in sources vector).
     *
     * @return index of the selected source
     */
    int getSelectedViewerSourceIdx () const { return selectedViewer->sourceIdx; }

    /** 
     * Getter to the previously selected viewer's viewer widget.
     *
     * @return pointer to the viewer widget of the previously selected viewer
     */
    M4D::Viewer::m4dGUIAbstractViewerWidget *getPrevSelectedViewerWidget () const { return prevSelectedViewer->viewerWidget; }
   
    Imaging::ConnectionInterface *getDefaultConnection () const { return defaultConnection; }


    /** 
     * Setter for the selected viewer's checked tool (index) - for left mouse button.
     *
     * @param value value to set (the index of the tool)
     */
    void setSelectedViewerLeftTool ( unsigned value ) { selectedViewer->checkedLeftButtonTool = value; }

    /** 
     * Setter for the selected viewer's checked tool (index) - for right mouse button.
     *
     * @param value value to set (the index of the tool)
     */
    void setSelectedViewerRightTool ( unsigned value ) { selectedViewer->checkedRightButtonTool = value; }

    void setDefaultConnection ( Imaging::ConnectionInterface *conn );

    void setConnectionForAll ( Imaging::ConnectionInterface *conn );

    void setViewerEventHandlerForSelected ( M4D::Viewer::m4dGUIViewerEventHandlerInterface *eventHandler );

    void setViewerEventHandlerForAll ( M4D::Viewer::m4dGUIViewerEventHandlerInterface *eventHandler );


    /** 
     * Raplces the selected viewer - keeping the connection, overlay infos, etc. from the replaced viewer.
     * It's inserted to the repleced viewer's place, with its dimensions. The whole toolBar and the controls
     * are updated according to the type of the new viewer.
     *
     * @param viewerFactory pointer to the Viewer Factory - which will create the new viewer
     * @param replacedViewer pointer to the replaced viewer
     */
    void replaceSelectedViewerWidget ( ViewerFactory *viewerFactory, M4D::Viewer::m4dGUIAbstractViewerWidget *replacedViewer );

    /** 
     * Adds source (pipeline connection) to vector of registered sources - possible connections, 
     * where can be plugged a viewer. Can be selected through comboBox in toolBar. 
     *
     * @param conn pointer to the connection to be added
     * @param viewerEventHandler pointer to event handler, which should be active on the viewer connected
     * to this source
     */
    void addSource ( Imaging::ConnectionInterface *conn, 
                     M4D::Viewer::m4dGUIViewerEventHandlerInterface *viewerEventHandler );

  private slots:

    /** 
     * Slot for changing the Desktop's layout - should be connected to Screen Layout Widget.
     *
     * @param rows number of rows in the new layout
     * @param columns number of columns in the new layout
     */
    void setDesktopLayout ( const unsigned rows, const unsigned columns );

    /** 
     * Slot for handling selected viewer changing.
     *
     * @param index index of currently selected viewer
     */
    void selectedChanged ( unsigned index );

    /** 
     * Slot for Source comboBox activity - should be connected to it's activated( int ) signal - it's
     * setting input port for selected viewer widget.
     *
     * @param index index of selected source (in comboBox and also in sources vector)
     */
    void sourceSelected ( int index );

  signals:

    /**
     * Signal indicating selected viewer change - should be connected to main window's 
     * features slot - to update the whole adaptable toolBar.
     *
     * @param prevViewer pointer to the previously selected viewer - to disconnect it
     */
    void propagateFeatures ( M4D::Viewer::m4dGUIAbstractViewerWidget *prevViewer );

  private:

    /// Default Viewer Factory - for creating viewers.
    ViewerFactory *viewerFactory;

    /// Vector of viewer structures in the layout.
    std::vector< Viewer * > viewers;
    /// Pointer to the currently selected viewer.
    Viewer *selectedViewer;
    /// Pointer to the previously selected viewer.
    Viewer *prevSelectedViewer;

    /// Connection which will be used for newly added viewers.
    Imaging::ConnectionInterface *defaultConnection;

    /**
     * Vector of registered sources - possible connections, where can be plugged a viewer. Can be selected
     * through comboBox in toolBar + event handlers, which should be active on the viewer connected
     * to particular source.
     */ 
    std::vector< Source > sources;

    /// Number of rows in the layout.
    unsigned layoutRows;
    /// Number of columns in the layout.
    unsigned layoutColumns;
};

} // namespace GUI
} // namespace M4D

#endif // M4D_GUI_MAIN_VIEWER_DESKTOP_H


/** @} */

