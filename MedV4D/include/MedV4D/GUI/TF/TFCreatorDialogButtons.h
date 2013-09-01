#ifndef TF_CREATORDIALOGBUTTONS
#define TF_CREATORDIALOGBUTTONS

#include <QtWidgets/QRadioButton>
#include <QtWidgets/QVBoxLayout>

#include "MedV4D/GUI/TF/TFCommon.h"

#include "MedV4D/GUI/TF/TFDimensions.h"
#include "MedV4D/GUI/TF/TFFunctions.h"
#include "MedV4D/GUI/TF/TFPainters.h"
#include "MedV4D/GUI/TF/TFModifiers.h"
#include "MedV4D/GUI/TF/TFPredefined.h"

namespace M4D {
namespace GUI {

class TFPredefinedDialogButton: public QRadioButton{

	Q_OBJECT

public:

	TFPredefinedDialogButton(TF::Types::Predefined type, QWidget* parent = 0):
		QRadioButton(parent),
		type_(type){

		bool dialogButtonConnected = QObject::connect( this, SIGNAL(toggled(bool)), this, SLOT(on_toggled(bool)));
		tfAssert(dialogButtonConnected);
	}
	~TFPredefinedDialogButton(){}

signals:

	void Activated(TF::Types::Predefined active);

private slots:

	void on_toggled(bool checked){

		if(checked) emit Activated(type_);
	}

private:

	TF::Types::Predefined type_;
};

class TFDimensionDialogButton: public QRadioButton{

	Q_OBJECT

public:

	TFDimensionDialogButton(TF::Types::Dimension type, QWidget* parent = 0):
		QRadioButton(parent),
		type_(type){

		bool dialogButtonConnected = QObject::connect( this, SIGNAL(toggled(bool)), this, SLOT(on_toggled(bool)));
		tfAssert(dialogButtonConnected);
	}
	~TFDimensionDialogButton(){}

signals:

	void Activated(TF::Types::Dimension active);

private slots:

	void on_toggled(bool checked){

		if(checked) emit Activated(type_);
	}

private:

	TF::Types::Dimension type_;
};

class TFFunctionDialogButton: public QRadioButton{

	Q_OBJECT

public:

	TFFunctionDialogButton(TF::Types::Function type, QWidget* parent = 0):
		QRadioButton(parent),
		type_(type){

		bool dialogButtonConnected = QObject::connect( this, SIGNAL(toggled(bool)), this, SLOT(on_toggled(bool)));
		tfAssert(dialogButtonConnected);
	}
	~TFFunctionDialogButton(){}

signals:

	void Activated(TF::Types::Function active);

private slots:

	void on_toggled(bool checked){

		if(checked) emit Activated(type_);
	}

private:

	TF::Types::Function type_;
};

class TFPainterDialogButton: public QRadioButton{

	Q_OBJECT

public:

	TFPainterDialogButton(TF::Types::Painter type, QWidget* parent = 0):
		QRadioButton(parent),
		type_(type){

		bool dialogButtonConnected = QObject::connect( this, SIGNAL(toggled(bool)), this, SLOT(on_toggled(bool)));
		tfAssert(dialogButtonConnected);
	}
	~TFPainterDialogButton(){}

signals:

	void Activated(TF::Types::Painter active);

private slots:

	void on_toggled(bool checked){

		if(checked) emit Activated(type_);
	}

private:

	TF::Types::Painter type_;
};

class TFModifierDialogButton: public QRadioButton{

	Q_OBJECT

public:

	TFModifierDialogButton(TF::Types::Modifier type, QWidget* parent = 0):
		QRadioButton(parent),
		type_(type){

		bool dialogButtonConnected = QObject::connect( this, SIGNAL(toggled(bool)), this, SLOT(on_toggled(bool)));
		tfAssert(dialogButtonConnected);
	}
	~TFModifierDialogButton(){}

signals:

	void Activated(TF::Types::Modifier active);

private slots:

	void on_toggled(bool checked){

		if(checked) emit Activated(type_);
	}

private:

	TF::Types::Modifier type_;
};

} // namespace GUI
} // namespace M4D

#endif	//TF_CREATORDIALOGBUTTONS
