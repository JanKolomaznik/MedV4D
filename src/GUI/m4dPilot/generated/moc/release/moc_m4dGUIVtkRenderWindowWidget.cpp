/****************************************************************************
** Meta object code from reading C++ file 'm4dGUIVtkRenderWindowWidget.h'
**
** Created: Sat Apr 19 23:05:21 2008
**      by: The Qt Meta Object Compiler version 59 (Qt 4.3.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../widgets/m4dGUIVtkRenderWindowWidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'm4dGUIVtkRenderWindowWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 59
#error "This file was generated using the moc from 4.3.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

static const uint qt_meta_data_m4dGUIVtkRenderWindowWidget[] = {

 // content:
       1,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets

       0        // eod
};

static const char qt_meta_stringdata_m4dGUIVtkRenderWindowWidget[] = {
    "m4dGUIVtkRenderWindowWidget\0"
};

const QMetaObject m4dGUIVtkRenderWindowWidget::staticMetaObject = {
    { &QVTKWidget::staticMetaObject, qt_meta_stringdata_m4dGUIVtkRenderWindowWidget,
      qt_meta_data_m4dGUIVtkRenderWindowWidget, 0 }
};

const QMetaObject *m4dGUIVtkRenderWindowWidget::metaObject() const
{
    return &staticMetaObject;
}

void *m4dGUIVtkRenderWindowWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_m4dGUIVtkRenderWindowWidget))
	return static_cast<void*>(const_cast< m4dGUIVtkRenderWindowWidget*>(this));
    return QVTKWidget::qt_metacast(_clname);
}

int m4dGUIVtkRenderWindowWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QVTKWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
