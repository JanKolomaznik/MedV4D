#ifndef M4D_GUI_MAIN_WINDOW_H
#define M4D_GUI_MAIN_WINDOW_H

#include <QMainWindow>

#include "GUI/m4dGUIVtkRenderWindowWidget.h"
#include "GUI/m4dGUIStudyManagerWidget.h"


class m4dGUIMainWindow: public QMainWindow
{
  Q_OBJECT

  public:
    m4dGUIMainWindow ();

  private slots:
    void open ();
    void search ();

  private:
    void createStudyManagerDialog ();
    void createVtkRenderWindowWidget ();
    
    void createActions ();
    void createMenus ();
    void createToolBars ();
    void createStatusBar ();
    void createDockWindows();

    QWidget *centralWidget;
    m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget;

    QDialog *studyManagerDialog;
  
    QAction *searchAct;
    QAction *openAct;
    QAction *saveAct;
    QAction *exitAct;

    QToolBar *fileToolBar;
    QToolBar *searchToolBar;
   
    QMenu *fileMenu;
    QMenu *viewMenu;
};

#endif // M4D_GUI_MAIN_WINDOW_H