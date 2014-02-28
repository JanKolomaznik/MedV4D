#ifndef TF_CREATORDIALOGBUTTONS
#define TF_CREATORDIALOGBUTTONS

#include <QtWidgets/QRadioButton>
#include <QtWidgets/QVBoxLayout>

#include "MedV4D/GUI/TF/Common.h"

#include "MedV4D/GUI/TF/Dimensions.h"
#include "MedV4D/GUI/TF/Functions.h"
#include "MedV4D/GUI/TF/Painters.h"
#include "MedV4D/GUI/TF/Modifiers.h"
#include "MedV4D/GUI/TF/Predefined.h"

namespace M4D {
namespace GUI {

class PredefinedDialogButton: public QRadioButton{

	Q_OBJECT

public:

	PredefinedDialogButton(TF::Types::Predefined type, QWidget* parent = 0):
		QRadioButton(parent),
		type_(type){

		bool dialogButtonConnected = QObject::connect( this, SIGNAL(toggled(bool)), this, SLOT(on_toggled(bool)));
		tfAssert(dialogButtonConnected);
	}
	~PredefinedDialogButton(){}

signals:

	void Activated(TF::Types::Predefined active);

private slots:

	void on_toggled(bool checked){

		if(checked) emit Activated(type_);
	}

private:

	TF::Types::Predefined type_;
};

class DimensionDialogButton: public QRadioButton{

	Q_OBJECT

public:

	DimensionDialogButton(TF::Types::Dimension type, QWidget* parent = 0):
		QRadioButton(parent),
		type_(type){

		bool dialogButtonConnected = QObject::connect( this, SIGNAL(toggled(bool)), this, SLOT(on_toggled(bool)));
		tfAssert(dialogButtonConnected);
	}
	~DimensionDialogButton(){}

signals:

	void Activated(TF::Types::Dimension active);

private slots:

	void on_toggled(bool checked){

		if(checked) emit Activated(type_);
	}

private:

	TF::Types::Dimension type_;
};

class FunctionDialogButton: public QRadioButton{

	Q_OBJECT

public:

	FunctionDialogButton(TF::Types::Function type, QWidget* parent = 0):
		QRadioButton(parent),
		type_(type){

		bool dialogButtonConnected = QObject::connect( this, SIGNAL(toggled(bool)), this, SLOT(on_toggled(bool)));
		tfAssert(dialogButtonConnected);
	}
	~FunctionDialogButton(){}

signals:

	void Activated(TF::Types::Function active);

private slots:

	void on_toggled(bool checked){

		if(checked) emit Activated(type_);
	}

private:

	TF::Types::Function type_;
};

class PainterDialogButton: public QRadioButton{

	Q_OBJECT

public:

	PainterDialogButton(TF::Types::Painter type, QWidget* parent = 0):
		QRadioButton(parent),
		type_(type){

		bool dialogButtonConnected = QObject::connect( this, SIGNAL(toggled(bool)), this, SLOT(on_toggled(bool)));
		tfAssert(dialogButtonConnected);
	}
	~PainterDialogButton(){}

signals:

	void Activated(TF::Types::Painter active);

private slots:

	void on_toggled(bool checked){

		if(checked) emit Activated(type_);
	}

private:

	TF::Types::Painter type_;
};

class ModifierDialogButton: public QRadioButton{

	Q_OBJECT

public:

	ModifierDialogButton(TF::Types::Modifier type, QWidget* parent = 0):
		QRadioButton(parent),
		type_(type){

		bool dialogButtonConnected = QObject::connect( this, SIGNAL(toggled(bool)), this, SLOT(on_toggled(bool)));
		tfAssert(dialogButtonConnected);
	}
	~ModifierDialogButton(){}

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
