/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file m4dGUIMainViewerDesktopWidget.h 
 * @{ 
 **/

#ifndef M4D_GUI_MAIN_VIEWER_DESKTOP_H
#define M4D_GUI_MAIN_VIEWER_DESKTOP_H

#include <QtGui>

#include "GUI/m4dGUIAbstractViewerWidget.h"
#include "GUI/m4dGUISliceViewerWidget.h"
#include "GUI/m4dGUIVtkViewerWidget.h"


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
     * Enumeration for concrete viewer type - viewers derived from m4dGUIAbstractViewerWidget.
     */
    typedef enum { SLICE_VIEWER, VTK_VIEWER } ViewerType;

    /**
     * Structure representing a viewer (it contains its concrete type, the viewer widget, which 
     * tools were checked for it, index of its source - pipeline connection).
     */
    struct Viewer {
      /// Concrete type of the viewer.
      ViewerType type;
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
     * Main Viewer Desktop constructor.
     *
     * @param parent pointer to the parent widget - default is 0
     */
    m4dGUIMainViewerDesktopWidget ( QWidget *parent = 0 );

    /** 
     * Getter to the selected viewer's type.
     *
     * @return type of the selected viewer
     */
    ViewerType getSelectedViewerType () const { return selectedViewer->type; }

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
     * Raplces the selected viewer - keeping the connection, overlay infos, etc. from the replaced viewer.
     * It's inserted to the repleced viewer's place, with its dimensions. The whole toolBar and the controls
     * are updated according to the type of the new viewer.
     *
     * @param type type of the wanted viewer
     * @param replacedViewer pointer to the replaced viewer
     */
    void replaceSelectedViewerWidget ( ViewerType type, M4D::Viewer::m4dGUIAbstractViewerWidget *replacedViewer );
    

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


    /** 
     * Adds source (pipeline connection) to vector of registered sources - possible connections, 
     * where can be plugged a viewer. Can be selected through comboBox in toolBar. 
     *
     * @param conn pointer to the connection to be added
     * @param pipelineDescription description/name of the pipeline connection belongs to (for the user - in the comboBox)
     * @param connectionDescription description of the connection (for the user - in the comboBox)
     */
    void addSource ( Imaging::ConnectionInterface *conn, const char *pipelineDescription,
                     const char *connectionDescription );

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
     */
    void propagateFeatures ();

    /**
     * Signal indicating source (pipeline connection) addition - it's emitted after the addition.
     * Should be connected to main window's source slot - to add item to sources toolBar.
     */
    void sourceAdded ( const QString &pipelineDescription, const QString &connectionDescription );

  private:

    /// Vector of viewer structures in the layout.
    std::vector< Viewer * > viewers;
    /// Pointer to the currently selected viewer.
    Viewer *selectedViewer;
    /// Pointer to the previously selected viewer.
    Viewer *prevSelectedViewer;

    /// Connection which will be used for newly added viewers
    Imaging::ConnectionInterface *defaultConnection;

    /**
     * Vector of registered sources - possible connections, where can be plugged a viewer. Can be selected
     * through comboBox in toolBar. 
     */ 
    std::vector< Imaging::ConnectionInterface * > sources;

    /// Number of rows in the layout.
    unsigned layoutRows;
    /// Number of columns in the layout.
    unsigned layoutColumns;
};

} // namespace GUI
} // namespace M4D

#endif // M4D_GUI_MAIN_VIEWER_DESKTOP_H


/** @} */

