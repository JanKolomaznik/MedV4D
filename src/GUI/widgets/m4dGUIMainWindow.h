/**
 * @ingroup gui 
 * @author Attila Ulman 
 * @file m4dGUIMainWindow.h 
 *
 * @{ 
 * 
 *  The MedV4D project uses the Qt library to create and manipulate GUI
 *  elements. Qt is a very powerful, object oriented,	cross platform GUI
 *  library. Since it has an X Window System implementation as well as a 
 *  Windows implementation,	it allows the software to run under Windows 
 *  and Unices as well. It works together very well with many other useful
 *  toolkits,	one of which is the VTK (Visualization Toolkit) toolkit, 
 *  which is also used as an optional component of the software.
 **/

#ifndef M4D_GUI_MAIN_WINDOW_H
#define M4D_GUI_MAIN_WINDOW_H

#include <QMainWindow>
#include <QtGui>

#include "m4dGUIMainViewerDesktopWidget.h"
#include "m4dGUIStudyManagerWidget.h"
#include "m4dGUIToolBarCustomizerWidget.h"
#include "m4dGUIScreenLayoutWidget.h"
#include "m4dGUIProgressBarWidget.h"


namespace M4D {
namespace GUI {

/// ID of actions and also ordering in their array
#define ACTION_EMPTY             0
#define ACTION_WINDOW_LEVEL      1
#define ACTION_PAN               2
#define ACTION_ZOOM              3
#define ACTION_STACK             4
#define ACTION_OVERLAY           5
#define ACTION_PROBE             6
#define ACTION_NEW_POINT         7
#define ACTION_NEW_SHAPE         8
#define ACTION_CLEAR_POINT       9
#define ACTION_CLEAR_SHAPE       10
#define ACTION_CLEAR_ALL         11
#define ACTION_FLIP_HORIZONTAL   12
#define ACTION_FLIP_VERTICAL     13
#define ACTION_SLICE_ORIENTATION 14
#define ACTION_ROTATE_3D         15

/**
 * Class representing the Main Window - containing all basic dialogs, viewer desktop
 * with various viewers and layout managing, adaptive toolBars, menus - all with uniform look.
 *
 * Specific applications should derive from this class and reimplement it's methods (e.g. provess) 
 * or extend it's functionality (e.g. add new widgets, pipeline connections).
 */
class m4dGUIMainWindow: public QMainWindow
{
  Q_OBJECT

  public:

    /**
     * Enumeration for type of the dockWindow (positioning) - which can be added to the mainWindow for
     * custom pipeline/filter settings widgets.
     */
    typedef enum { FLOATING_DOCK_WINDOW, DOCKED_DOCK_WINDOW } DockWindowType;

    /**
     * Enumeration for type of the tool - toolBar behavior: checkable tool, toggle tool.
     */
    typedef enum { CHECKABLE_TOOL, TOGGLE_TOOL } ToolType;

    /**
     * Enumeration for type of the mouse button - left, right.
     */
    typedef enum { LEFT_BUTTON, RIGHT_BUTTON } ButtonType;

    /**
     * Structure representing a study (it contains pointer to its DicomObjSet and overlay info lists).
     * Study Manager, loading method are filling it.
     */
    struct Study {
      /// Pointer to DicomObjSet.
      Dicom::DicomObjSet *dicomObjSet;
      /// Overlay info lists.
      std::list< std::string > leftOverlayInfo, rightOverlayInfo;
    };


    /** 
     * Main Window constructor.
     *
     * @param appName name of the application - for title and QSettings prefix
     * @param orgName name of the organization - for QSettings prefix
     * @param icon reference to the icon of the main window, application - default
     * is universal app.png from incons directory
     */
    m4dGUIMainWindow ( const char *appName, const char *orgName,
                       const QIcon &icon = QIcon( ":/icons/app.png" ) );


    /**
     * Returns flag indicating wheather the build was successful - construction of the 
     * Study Manager Widget, especially its Study List Component (DcmProvider construction)
     * can cause exceptions (e.g. for missing config file).
     *
     * @return build success flag
     */
    bool wasBuildSuccessful () const { return studyManagerWidget->getStudyListComponent()->wasBuidSuccessful(); }

    /**
     * Returns the build message - in case of problems - it's filled with text
     * of the build exception. Construction of the Study Manager Widget, especially its Study List 
     * Component (DcmProvider construction) can cause exceptions (e.g. for missing config file).
     *
     * @return build message string
     */
    QString getBuildMessage () const { return studyManagerWidget->getStudyListComponent()->getBuildMessage(); }


    /** 
     * Adds a widget to the Stacking Desktop - can be anything derived from QWidget. 
     * If there is already one added, will be replaced.
     * This method is just adding the widget, current widget will be kept - no switching.
     * 
     * @param widget pointer to the widget, which should be added
     */
    void addDesktopWidget ( QWidget *widget );

    /** 
     * Adds a Viewer Desktop Widget to the Stacking Desktop.  
     * Can be added as many as needed.
     * This method is just adding the widget, current widhet will be kept - no switching.
     * 
     * @param viewerDesktop pointer to the Viewer Desktop Widget, which should be added
     * @return index in the Stacking Desktop - used for swithing to this desktop
     */
    int addViewerDesktop ( m4dGUIMainViewerDesktopWidget *viewerDesktop );

    /** 
     * Switches to a previously added widget - in the Stacking Desktop. 
     * Causes toolBar and control adaptations.
     */
    void switchToDesktopWidget ();

    /** 
     * Switches to a previously added Viewer Desktop Widget - in the Stacking Desktop. 
     * Causes toolBar and control adaptations.
     * 
     * @param index index in the Stacking Desktop - returned by addViewerDesktop
     */
    void switchToViewerDesktop ( int index );

    /** 
     * Switches to the default Viewer Desktop Widget - in Stacking Desktop. 
     * Causes toolBar and control adaptations.
     */
    void switchToDefaultViewerDesktop ();

    /** 
     * Adds source (pipeline connection) to vector of registered sources - possible connections, 
     * where can be plugged a viewer. Can be selected through comboBox in toolBar.
     * It's calling the Main Viewer Desktop's addSource method.
     *
     * @param conn pointer to the connection to be added
     * @param pipelineDescription description/name of the pipeline connection belongs to (for the user - in the comboBox)
     * @param connectionDescription description of the connection (for the user - in the comboBox)
     */
    void addSource ( Imaging::ConnectionInterface *conn, const char *pipelineDescription,
                     const char *connectionDescription );

    /** 
     * Adds dockWindow (widget) to the application - e.g. for pipeline/filter settings.
     * It handles its position, size, hide/show menu items, etc.
     * 
     * @param title title of the dockWindow
     * @param widget pointer to the widget, which should be in the added dockWindow
     * @param windowType type of the dockWindow
     */
    void addDockWindow ( const char *title, QWidget *widget, DockWindowType windowType = FLOATING_DOCK_WINDOW );

  private slots:

    /**
     * Slot for search - to show the Study Manager. After closing it (clicking on View) it calls 
     * process method.
     */
    void search ();

    /**
     * Slot for open - to show the Open Dialog - open DICOM file from disk.
     */
    void open ();

    /**
     * Slot for load - to show the Load Dialog - load saved Data Set from disk.
     */
    void load ();

    /**
     * Slot for save - to show the Save Dialog - save current Data Set to disk.
     */
    void save ();

    /**
     * Slot for toolBar customization - to show the ToolBar Customizer Dialog.
     */
    void customize ();

    /**
     * Slot for managing ToolBar sizes.
     */
    void size ();

    /**
     * Slot for Status Bar - to show or hide it.
     */
    void status ();

    /**
     * Slots for setting controls and handlers for tools. Indirect slots calling delegateAction method.
     * For each checkable tool one specific slot (toggle tools are direct to viewers).
     */
    void viewerWindowLevel ();
    void viewerPan ();
    void viewerZoom ();
    void viewerStack ();
    void viewerProbe ();
    void viewerNewPoint ();
    void viewerNewShape ();
    void viewerRotate ();

    /**
     * Slot for managing adaptive toolBars depending on the type of the selected viewer.
     * Performing reconnections, updates, triggers, etc.
     * 
     * @param prevViewer pointer to the previously selected viewer - to disconnect it
     */
    void features ( M4D::Viewer::m4dGUIAbstractViewerWidget *prevViewer );

    /**
     * Slot for layout settings - to show Screen Layout Dialog.
     */
    void layout ();

    /**
     * Slot for replacing selected viewers (toggle 3D viewer) - with updates (features).
     */
    void replace ();

    /**
     * Slot for managing sources toolBar behavior - adding new sources - connected to Main Viewer Desktop.
     */
    void source ( const QString &pipelineDescription, const QString &connectionDescription );

    void startProgress ();

    void stopProgress ();

    void loadingReady ();

    void loadingException ( const QString &description );

  signals:

    /**
     * Signal indicating checkable (left, right - exl.) tools change -> change the handlers, buttons accordingly.
     *
     * @param hnd button handler for the selected viewer (used in its slot)
     * @param btn type of the button for the selected viewer (used in its slot)
     */
    void toolChanged ( Viewer::m4dGUIAbstractViewerWidget::ButtonHandler hnd, 
                       Viewer::m4dGUIAbstractViewerWidget::MouseButton btn );

  private:

    /**
     * Creates default Viewer Desktop and connects it with other widgets.
     */
    void createDefaultViewerDesktop ();

    /**
     * Creates Study Manager Dialog with Study Manager Widget and connects it with other widgets.
     */
    void createStudyManagerDialog ();

    /**
     * Creates ToolBar Customizer Dialog with ToolBar Customizer Widget and connects it with other widgets.
     */
    void createToolBarCustomizerDialog ();

    /**
     * Creates Screen Layout Dialog with Screen Layout Widget and connects it with other widgets.
     */
    void createScreenLayoutDialog ();

    /**
     * Creates Progress Bar Dialog with Progress Bar Widget.
     */
    void createProgressBarDialog ();
        

    /**
     * Creates all the actions - used in toolBars, menus, etc.
     */
    void createActions ();

    /**
     * Creates menus - from actions.
     */
    void createMenus ();

    /**
     * Creates toolBars - from actions & sources comboBox.
     */
    void createToolBars ();

    /**
     * Sets up the statusBar.
     */
    void createStatusBar ();


    /** 
     * Connects to specific slot of the selected viewer - to set controls and handlers for tools.
     * Indirect connections.
     * 
     * @param actionIdx order in viewerActs (macro for each tool)
     * @param hnd button handler for the selected viewer (used in its slot)
     */
    void delegateAction ( unsigned actionIdx, Viewer::m4dGUIAbstractViewerWidget::ButtonHandler hnd );

  protected:

    /** 
     * Virtual method - called after closing the Study Manager (clicking on View) to
     * process the result of the Study Manager (search). Default behavor: creates
     * image from the Dicom Object Set and sets it as input of the currently selected viewer.
     * 
     * Can be reimplemented to get specific behavior.
     * 
     * @param dicomObjSet Dicom Object Set (result of the search) to process
     */
    virtual void process ( Dicom::DicomObjSetPtr dicomObjSet );


    /// Pointer to the current Viewer Desktop
    m4dGUIMainViewerDesktopWidget *currentViewerDesktop;
    /// Pointer to the Study Manager Widget
    m4dGUIStudyManagerWidget *studyManagerWidget;
    /// Pointer to the ToolBar Customizer Widget
    m4dGUIToolBarCustomizerWidget *toolBarCustomizerWidget;
    /// Pointer to the Screen Layout Widget
    m4dGUIScreenLayoutWidget *screenLayoutWidget;
    /// Pointer to the Progress Bar Widget
    m4dGUIProgressBarWidget *progressBarWidget;
  
  private:

    /// Names of the action icons.
    static const char *actionIconNames[];
    /// Texts of the actions.
    static const char *actionTexts[];
    /// Tool types of the actions - information also used for weather the connection is direct to the viewer.
    static const ToolType   actionToolTypes[];
    /// Types of the mouse button for the actions (checkable tools).
    static const ButtonType actionButtonTypes[];
    /// Shortcuts of the actions.
    static const char *actionShortCuts[];
    /// StatusTips of the actions.
    static const char *actionStatusTips[];
    /// Slots for the actions.
    static const char *actionSlots[];
    /// Mapping available slots of viewers to actions.
    static const int   slotsToActions[]; 
  
    /// Dialogs for widgets.
    QDialog *studyManagerDialog;
    QDialog *toolBarCustomizerDialog;
    QDialog *screenLayoutDialog;
    QDialog *progressBarDialog;
  
    /// Actions.
    QAction *searchAct;
    QAction *openAct;
    QAction *loadAct;
    QAction *saveAct;
    QAction *exitAct;
    QAction *customizeAct;
    QAction *smallAct;
    QAction *mediumAct;
    QAction *largeAct;
    QAction *statusAct;
    QAction **viewerActs;
    QAction *layoutAct;
    QAction *replaceAct;
    
    /// ToolBars.
    QToolBar *searchToolBar;
    QToolBar *fileToolBar;
    QToolBar *layoutToolBar;
    QToolBar *viewerToolBar;
    QToolBar *replaceToolBar;
    QToolBar *sourcesToolBar;

    QStackedWidget *mainDesktopStackedWidget;

    /**
     * ComboBox in toolBar for selecting from registered sources - possible connections, where can
     * be plugged a viewer. It's visible only if there is registered at least one connection.
     */
    QComboBox *sourcesComboBox;
 
    /// Menus.
    QMenu *fileMenu;
    QMenu *toolBarsMenu;
    QMenu *toolsMenu;
    QMenu *viewMenu;

    /// Current study - Study Manager, loading method are filling it.
    Study actualStudy;

    /// Current path - history of Open/Load dialogs
    QString currentOpenPath, currentLoadPath;
};

} // namespace GUI
} // namespace M4D

#endif // M4D_GUI_MAIN_WINDOW_H



/** @} */

