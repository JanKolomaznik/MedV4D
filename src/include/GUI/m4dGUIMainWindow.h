#ifndef M4D_GUI_MAIN_WINDOW_H
#define M4D_GUI_MAIN_WINDOW_H

#include <QMainWindow>

#include "GUI/m4dGUIVtkRenderWindowWidget.h"


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
    void createDockWindows();

    QWidget *centralWidget;
    m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget;

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


