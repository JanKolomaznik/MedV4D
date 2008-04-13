#ifndef M4D_GUI_MAIN_WINDOW_H
#define M4D_GUI_MAIN_WINDOW_H

#include <QtGui/QMainWindow>

#include "m4dGUIVtkRenderWindowWidget.h"


class m4dGUIMainWindow: public QMainWindow
{
  Q_OBJECT

  public:
    m4dGUIMainWindow ();

  private slots:
     void open ();
     void search ();

  private:
     void createVtkRenderWindowWidget ();
     void createActions ();
     void createMenus ();

     QWidget *centralWidget;
     m4dGUIVtkRenderWindowWidget *vtkRenderWindowWidget;
  
     QAction *searchAct;
     QAction *openAct;
     QAction *exitAct;

     QMenu *fileMenu;
};

#endif // M4D_GUI_MAIN_WINDOW_H