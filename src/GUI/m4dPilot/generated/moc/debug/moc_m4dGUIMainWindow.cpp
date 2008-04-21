/****************************************************************************
** Meta object code from reading C++ file 'm4dGUIMainWindow.h'
**
** Created: Sun Apr 20 17:09:09 2008
**      by: The Qt Meta Object Compiler version 59 (Qt 4.3.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../widgets/m4dGUIMainWindow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'm4dGUIMainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 59
#error "This file was generated using the moc from 4.3.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

static const uint qt_meta_data_m4dGUIMainWindow[] = {

 // content:
       1,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   10, // methods
       0,    0, // properties
       0,    0, // enums/sets

 // slots: signature, parameters, type, tag, flags
      18,   17,   17,   17, 0x08,
      25,   17,   17,   17, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_m4dGUIMainWindow[] = {
    "m4dGUIMainWindow\0\0open()\0search()\0"
};

const QMetaObject m4dGUIMainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_m4dGUIMainWindow,
      qt_meta_data_m4dGUIMainWindow, 0 }
};

const QMetaObject *m4dGUIMainWindow::metaObject() const
{
    return &staticMetaObject;
}

void *m4dGUIMainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_m4dGUIMainWindow))
	return static_cast<void*>(const_cast< m4dGUIMainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int m4dGUIMainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: open(); break;
        case 1: search(); break;
        }
        _id -= 2;
    }
    return _id;
}
