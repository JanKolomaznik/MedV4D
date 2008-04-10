/********************************************************************************
** Form generated from reading ui file 'qtVtkMainWindow.ui'
**
** Created: Tue Apr 8 18:24:52 2008
**      by: Qt User Interface Compiler version 4.3.3
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef UI_QTVTKMAINWINDOW_H
#define UI_QTVTKMAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHBoxLayout>
#include <QtGui/QMainWindow>
#include <QtGui/QMenuBar>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
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
    QRadioButton *SphereRadio;
    QRadioButton *DicomRadio;
    QSpacerItem *spacerItem;
    QHBoxLayout *hboxLayout1;
    QSpacerItem *spacerItem1;
    QPushButton *CloseButton;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
    if (MainWindow->objectName().isEmpty())
        MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
    MainWindow->resize(556, 418);
    centralwidget = new QWidget(MainWindow);
    centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
    vboxLayout = new QVBoxLayout(centralwidget);
    vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
    qvtkWidget = new QVTKWidget(centralwidget);
    qvtkWidget->setObjectName(QString::fromUtf8("qvtkWidget"));

    vboxLayout->addWidget(qvtkWidget);

    hboxLayout = new QHBoxLayout();
    hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
    SphereRadio = new QRadioButton(centralwidget);
    SphereRadio->setObjectName(QString::fromUtf8("SphereRadio"));
    SphereRadio->setChecked(true);

    hboxLayout->addWidget(SphereRadio);

    DicomRadio = new QRadioButton(centralwidget);
    DicomRadio->setObjectName(QString::fromUtf8("DicomRadio"));
    DicomRadio->setChecked(false);

    hboxLayout->addWidget(DicomRadio);

    spacerItem = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

    hboxLayout->addItem(spacerItem);


    vboxLayout->addLayout(hboxLayout);

    hboxLayout1 = new QHBoxLayout();
    hboxLayout1->setObjectName(QString::fromUtf8("hboxLayout1"));
    spacerItem1 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

    hboxLayout1->addItem(spacerItem1);

    CloseButton = new QPushButton(centralwidget);
    CloseButton->setObjectName(QString::fromUtf8("CloseButton"));
    CloseButton->setDefault(true);

    hboxLayout1->addWidget(CloseButton);


    vboxLayout->addLayout(hboxLayout1);

    MainWindow->setCentralWidget(centralwidget);
    menubar = new QMenuBar(MainWindow);
    menubar->setObjectName(QString::fromUtf8("menubar"));
    menubar->setGeometry(QRect(0, 0, 556, 21));
    MainWindow->setMenuBar(menubar);
    statusbar = new QStatusBar(MainWindow);
    statusbar->setObjectName(QString::fromUtf8("statusbar"));
    MainWindow->setStatusBar(statusbar);

    retranslateUi(MainWindow);
    QObject::connect(CloseButton, SIGNAL(clicked()), MainWindow, SLOT(close()));
    QObject::connect(SphereRadio, SIGNAL(toggled(bool)), qvtkWidget, SLOT(update()));
    QObject::connect(DicomRadio, SIGNAL(toggled(bool)), qvtkWidget, SLOT(update()));

    QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
    MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0, QApplication::UnicodeUTF8));
    SphereRadio->setText(QApplication::translate("MainWindow", "Sphere", 0, QApplication::UnicodeUTF8));
    DicomRadio->setText(QApplication::translate("MainWindow", "DICOM", 0, QApplication::UnicodeUTF8));
    CloseButton->setText(QApplication::translate("MainWindow", "Close", 0, QApplication::UnicodeUTF8));
    Q_UNUSED(MainWindow);
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

#endif // UI_QTVTKMAINWINDOW_H
