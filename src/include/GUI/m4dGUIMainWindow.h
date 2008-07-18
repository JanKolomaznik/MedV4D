#ifndef M4D_GUI_MAIN_WINDOW_H
#define M4D_GUI_MAIN_WINDOW_H

#include <QMainWindow>
#include <QtGui>

#include "GUI/m4dGUIMainViewerDesktopWidget.h"
#include "GUI/m4dGUIStudyManagerWidget.h"
#include "GUI/m4dGUIScreenLayoutWidget.h"


class m4dGUIMainWindow: public QMainWindow
{
  Q_OBJECT

  public:
    m4dGUIMainWindow ( const char *title, const QIcon &icon = QIcon( ":/icons/app.png" ) );

  private slots:
    void search ();
    void open ();
    void layout ();
    void features ();

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
    QAction *layoutAct;
    QAction **viewerActs;
    
    QToolBar *searchToolBar;
    QToolBar *fileToolBar;
    QToolBar *layoutToolBar;
    QToolBar *viewerToolBar;
   
    QMenu *fileMenu;
    QMenu *viewMenu;  // name coll.
};

#endif // M4D_GUI_MAIN_WINDOW_H


