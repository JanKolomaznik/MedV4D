/****************************************************************************
** Meta object code from reading C++ file 'StManagerFilterComp.h'
**
** Created: Mon Apr 21 12:06:16 2008
**      by: The Qt Meta Object Compiler version 59 (Qt 4.3.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../widgets/components/StManagerFilterComp.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'StManagerFilterComp.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 59
#error "This file was generated using the moc from 4.3.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

static const uint qt_meta_data_StManagerFilterComp[] = {

 // content:
       1,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   10, // methods
       0,    0, // properties
       0,    0, // enums/sets

 // slots: signature, parameters, type, tag, flags
      21,   20,   20,   20, 0x08,
      30,   20,   20,   20, 0x08,
      42,   20,   20,   20, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_StManagerFilterComp[] = {
    "StManagerFilterComp\0\0search()\0fromCheck()\0"
    "toCheck()\0"
};

const QMetaObject StManagerFilterComp::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_StManagerFilterComp,
      qt_meta_data_StManagerFilterComp, 0 }
};

const QMetaObject *StManagerFilterComp::metaObject() const
{
    return &staticMetaObject;
}

void *StManagerFilterComp::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_StManagerFilterComp))
	return static_cast<void*>(const_cast< StManagerFilterComp*>(this));
    return QWidget::qt_metacast(_clname);
}

int StManagerFilterComp::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: search(); break;
        case 1: fromCheck(); break;
        case 2: toCheck(); break;
        }
        _id -= 3;
    }
    return _id;
}
