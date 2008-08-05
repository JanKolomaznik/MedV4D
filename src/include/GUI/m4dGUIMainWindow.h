#ifndef M4D_GUI_MAIN_WINDOW_H
#define M4D_GUI_MAIN_WINDOW_H

#include <QMainWindow>
#include <QtGui>

#include "GUI/m4dGUIMainViewerDesktopWidget.h"
#include "GUI/m4dGUIStudyManagerWidget.h"
#include "GUI/m4dGUIScreenLayoutWidget.h"


namespace M4D {
namespace GUI {

/// ID of actions and also ordering in their array
#define ACTION_EMPTY             0
#define ACTION_WINDOW_LEVEL      1
#define ACTION_PAN               2
#define ACTION_ZOOM              3
#define ACTION_STACK             4
#define ACTION_OVERLAY           5
#define ACTION_NEW_POINT         6
#define ACTION_NEW_SHAPE         7
#define ACTION_CLEAR_POINT       8
#define ACTION_CLEAR_SHAPE       9
#define ACTION_CLEAR_ALL         10
#define ACTION_FLIP_HORIZONTAL   11
#define ACTION_FLIP_VERTICAL     12
#define ACTION_ROTATE_3D         13

/**
 * Class representing the Main Window - containing all basic dialogs, viewer desktop
 * with various viewers and layout managing, adaptive toolBars, menus - all with uniform look.
 *
 * Specific applications should derive from this class and reimplement it's methods (e.g. view) 
 * or extend it's functionality (e.g. add new widgets).
 */
class m4dGUIMainWindow: public QMainWindow
{
  Q_OBJECT

  public:

    typedef enum { CHECKABLE_TOOL, TOGGLE_TOOL } ToolType;
    typedef enum { LEFT_BUTTON, RIGHT_BUTTON } ButtonType;

    /** 
     * Main Window constructor.
     *
     * @param title title of the main window, application
     * @param icon reference to the icon of the main window, application - default
     * is universal app.png from incons directory
     */
    m4dGUIMainWindow ( const char *title, const QIcon &icon = QIcon( ":/icons/app.png" ) );

    void addSource ( M4D::Imaging::ConnectionInterface *conn, const char *pipelineDescription,
                     const char *connectionDescription );
    void addDockWindow ( const char *title, QWidget *widget );

  private slots:

    /**
     * Slot for search - to show the Study Manager. After closing it (clicking on View) it calls 
     * view method.
     */
    void search ();

    /**
     * Slot for open - to show the Open Dialog - open DICOM file from disk.
     */
    void open ();

    /**
     * Slot for managing ToolBar sizes.
     */
    void size ();

    /**
     * Slot for Status Bar - to hide or show it.
     */
    void status ();

    /**
     * Slot for setting controls and handlers for tools. Indirect slots emiting signals to selected viewer.
     * For each checkable tool one specific slot.
     */
    void viewerWindowLevel ();
    void viewerPan ();
    void viewerZoom ();
    void viewerStack ();
    void viewerNewPoint ();
    void viewerNewShape ();
    void viewerRotate ();

    /**
     * Slot for managing adaptive toolBars depending on the type of the selected viewer.
     */
    void features ();

    /**
     * Slot for layout settings - to show Screen Layout Widget.
     */
    void layout ();

    /**
     * Slot for replacing selected viewers (toggle VTK viewer)
     */
    void replace ();

    void sources ( const QString &pipelineDescription, const QString &connectionDescription );

  signals:
    // clickable (left, right - exl.) tools changed -> change the handlers, buttons accordingly
    void toolChanged ( M4D::Viewer::m4dGUIAbstractViewerWidget::ButtonHandler, 
                       M4D::Viewer::m4dGUIAbstractViewerWidget::MouseButton );

  private:

    void createMainViewerDesktop ();
    void createStudyManagerDialog ();
    void createScreenLayoutDialog ();
        
    void createActions ();
    void createMenus ();
    void createToolBars ();
    void createStatusBar ();

    /** 
     * Connects to specific slot of the selected viewer - to set controls and handlers for tools.
     * Indirect connections.
     * 
     * @param actionIdx order in viewerActs (macro for each tool)
     * @param hnd button handler for the selected viewer (used in its slot)
     */
    void delegateAction ( unsigned actionIdx, M4D::Viewer::m4dGUIAbstractViewerWidget::ButtonHandler hnd );

    virtual void view ( M4D::Dicom::DcmProvider::DicomObjSet *dicomObjSet );

  protected:

    m4dGUIMainViewerDesktopWidget *mainViewerDesktop;
    m4dGUIStudyManagerWidget *studyManagerWidget;
    m4dGUIScreenLayoutWidget *screenLayoutWidget;
  
  private:

    static const char *actionIconNames[];
    static const char *actionTexts[];
    static const ToolType   actionToolTypes[];
    static const ButtonType actionButtonTypes[];
    static const char *actionShortCuts[];
    static const char *actionStatusTips[];
    static const char *actionSlots[];
    static const int   slotsToActions[]; 
  
    QDialog *studyManagerDialog;
    QDialog *screenLayoutDialog;
  
    QAction *searchAct;
    QAction *openAct;
    QAction *saveAct;
    QAction *exitAct;
    QAction *smallAct;
    QAction *mediumAct;
    QAction *largeAct;
    QAction *statusAct;
    QAction **viewerActs;
    QAction *layoutAct;
    QAction *replaceAct;
    
    QToolBar *searchToolBar;
    QToolBar *fileToolBar;
    QToolBar *layoutToolBar;
    QToolBar *viewerToolBar;
    QToolBar *replaceToolBar;
    QToolBar *sourcesToolBar;

    QComboBox *sourcesComboBox;
   
    QMenu *fileMenu;
    QMenu *toolBarsMenu;
    QMenu *toolsMenu;
    QMenu *viewMenu;

    
};

} // namespace GUI
} // namespace M4D

#endif // M4D_GUI_MAIN_WINDOW_H


