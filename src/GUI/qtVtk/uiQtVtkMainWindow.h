/********************************************************************************
** Form generated from reading ui file 'qtVtkMainWindow.ui'
**
** Created: Sun Mar 30 21:52:45 2008
**      by: Qt User Interface Compiler version 4.3.3
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef UIQTVTKMAINWINDOW_H
#define UIQTVTKMAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHBoxLayout>
#include <QtGui/QMainWindow>
#include <QtGui/QMenuBar>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QStatusBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "QVTKWidget.h"

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QVBoxLayout *vboxLayout;
    QVTKWidget *qvtkWidget;
    QHBoxLayout *hboxLayout;
    QSpacerItem *spacerItem;
    QPushButton *pushButton;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
    if (MainWindow->objectName().isEmpty())
        MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
    MainWindow->resize(520, 358);
    centralwidget = new QWidget(MainWindow);
    centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
    vboxLayout = new QVBoxLayout(centralwidget);
    vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
    qvtkWidget = new QVTKWidget(centralwidget);
    qvtkWidget->setObjectName(QString::fromUtf8("qvtkWidget"));

    vboxLayout->addWidget(qvtkWidget);

    hboxLayout = new QHBoxLayout();
    hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
    hboxLayout->setContentsMargins(-1, 8, -1, -1);
    spacerItem = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

    hboxLayout->addItem(spacerItem);

    pushButton = new QPushButton(centralwidget);
    pushButton->setObjectName(QString::fromUtf8("pushButton"));
    pushButton->setDefault(true);

    hboxLayout->addWidget(pushButton);


    vboxLayout->addLayout(hboxLayout);

    MainWindow->setCentralWidget(centralwidget);
    menubar = new QMenuBar(MainWindow);
    menubar->setObjectName(QString::fromUtf8("menubar"));
    menubar->setGeometry(QRect(0, 0, 520, 21));
    MainWindow->setMenuBar(menubar);
    statusbar = new QStatusBar(MainWindow);
    statusbar->setObjectName(QString::fromUtf8("statusbar"));
    MainWindow->setStatusBar(statusbar);

    retranslateUi(MainWindow);
    QObject::connect(pushButton, SIGNAL(clicked()), MainWindow, SLOT(close()));

    QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
    MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0, QApplication::UnicodeUTF8));
    pushButton->setText(QApplication::translate("MainWindow", "Close", 0, QApplication::UnicodeUTF8));
    Q_UNUSED(MainWindow);
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

#endif // UIQTVTKMAINWINDOW_H
