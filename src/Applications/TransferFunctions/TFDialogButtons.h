#ifndef TF_DIALOGBUTTONS
#define TF_DIALOGBUTTONS

#include <QtGui/QRadioButton>
#include <QtGui/QVBoxLayout>

#include <TFCommon.h>

#include <TFFunctions.h>
#include <TFPainters.h>
#include <TFModifiers.h>
#include <TFPredefined.h>

namespace M4D {
namespace GUI {

class TFFunctionDialogButton: public QRadioButton{

	Q_OBJECT

public:

	TFFunctionDialogButton(TF::Types::Function type, QWidget* parent = 0):
		QRadioButton(parent),
		type_(type){		

		bool dialogButtonConnected = QObject::connect( this, SIGNAL(toggled(bool)), this, SLOT(on_toggled(bool)));
		tfAssert(dialogButtonConnected);
	}

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

signals:

	void Activated(TF::Types::Modifier active);

private slots:

	void on_toggled(bool checked){			

		if(checked) emit Activated(type_);
	}

private:

	TF::Types::Modifier type_;
};

class TFPredefinedDialogButton: public QRadioButton{

	Q_OBJECT

public:

	TFPredefinedDialogButton(TF::Types::Predefined type, QWidget* parent = 0):
		QRadioButton(parent),
		type_(type){	

		bool dialogButtonConnected = QObject::connect( this, SIGNAL(toggled(bool)), this, SLOT(on_toggled(bool)));
		tfAssert(dialogButtonConnected);
	}

signals:

	void Activated(TF::Types::Predefined active);

private slots:

	void on_toggled(bool checked){			

		if(checked) emit Activated(type_);
	}

private:

	TF::Types::Predefined type_;
};

} // namespace GUI
} // namespace M4D

#endif	//TF_DIALOGBUTTONS