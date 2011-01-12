#include "TFAction.h"

namespace M4D {
namespace GUI {

TFAction::TFAction(QObject* parent, TFHolder::Type tfType):
	QAction(QString::fromStdString(convert<TFHolder::Type, std::string>(tfType)), parent),
	type_(tfType){

	bool tfActionEnabled = QObject::connect( this, SIGNAL(triggered()), this, SLOT(action_triggered()));
	tfAssert(tfActionEnabled);
}

TFAction::~TFAction(){
}

void TFAction::action_triggered(){

	emit TFActionClicked(type_);
}
} // namespace GUI
} // namespace M4D