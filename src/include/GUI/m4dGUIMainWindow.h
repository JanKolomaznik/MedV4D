#ifndef M4D_GUI_MAIN_WINDOW_H
#define M4D_GUI_MAIN_WINDOW_H

#include <QMainWindow>
#include <QtGui>

#include "GUI/m4dGUIMainViewerDesktopWidget.h"
#include "GUI/m4dGUIStudyManagerWidget.h"
#include "GUI/m4dGUIScreenLayoutWidget.h"


/**
 * @class m4dGUIMainWindow m4dGUIMainWindow.h
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

    /** 
     * Main Window constructor.
     *
     * @param title title of the main window, application
     * @param icon reference to the icon of the main window, application - default
     * is universal app.png from incons directory
     */
    m4dGUIMainWindow ( const char *title, const QIcon &icon = QIcon( ":/icons/app.png" ) );

  private slots:

    void search ();
    void open ();
    void size ();
    void status ();
    void viewerWindowLevel ();
    void viewerPan ();
    void viewerZoom ();
    void viewerStack ();
    void viewerNewPoint ();
    void viewerNewShape ();
    void features ();
    void layout (); 
    void swap ();

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
    void createDockWindows ();

    virtual void view ( M4D::Dicom::DcmProvider::DicomObjSet *dicomObjSet );

  protected:

    m4dGUIMainViewerDesktopWidget *mainViewerDesktop;
    m4dGUIStudyManagerWidget *studyManagerWidget;
    m4dGUIScreenLayoutWidget *screenLayoutWidget;
  
  private:

    static const char *actionIconNames[];
    static const char *actionTexts[];
    static const bool  actionCheckables[];
    static const bool  actionRightButtons[];
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
    QAction *swapAct;
    
    QToolBar *searchToolBar;
    QToolBar *fileToolBar;
    QToolBar *layoutToolBar;
    QToolBar *viewerToolBar;
    QToolBar *swapToolBar;
   
    QMenu *fileMenu;
    QMenu *toolBarsMenu;
    QMenu *toolsMenu;
    QMenu *viewMenu;
};

#endif // M4D_GUI_MAIN_WINDOW_H


