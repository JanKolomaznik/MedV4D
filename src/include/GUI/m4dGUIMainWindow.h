#ifndef M4D_GUI_MAIN_WINDOW_H
#define M4D_GUI_MAIN_WINDOW_H

#include <QMainWindow>

#include "GUI/m4dGUIVtkRenderWindowWidget.h"
#include "GUI/m4dGUIStudyManagerWidget.h"
#include "GUI/m4dGUIScreenLayoutWidget.h"


class m4dGUIMainWindow: public QMainWindow
{
  Q_OBJECT

  public:
    m4dGUIMainWindow ();

  private slots:
    void search ();
    void open ();
    void layout ();

  private:
    void createVtkRenderWindowWidget ();
    void createStudyManagerDialog ();
    void createScreenLayoutDialog ();
        
    void createActions ();
    void createMenus ();
    void createToolBars ();
    void createStatusBar ();
    void createDockWindows ();

    virtual void view ( M4D::Dicom::DcmProvider::DicomObjSet *dicomObjSet );

    QWidget *centralWidget;
    m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget;
    m4dGUIStudyManagerWidget *studyManagerWidget;
    m4dGUIScreenLayoutWidget *screenLayoutWidget;

    QDialog *studyManagerDialog;
    QDialog *screenLayoutDialog;
  
    QAction *searchAct;
    QAction *openAct;
    QAction *saveAct;
    QAction *exitAct;
    QAction *layoutAct;
    QAction *overlayAct;
    
    QToolBar *searchToolBar;
    QToolBar *fileToolBar;
    QToolBar *viewToolBar;
   
    QMenu *fileMenu;
    QMenu *viewMenu;  // name coll.
};

#endif // M4D_GUI_MAIN_WINDOW_H


