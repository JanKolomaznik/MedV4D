#include "TFCompositionDialog.h"

#include <TFPalette.h>
#include <TFBasicHolder.h>

namespace M4D {
namespace GUI {

TFCompositionDialog::TFCompositionDialog(TFPalette* palette):
	QDialog(palette),
	ui_(new Ui::TFCompositionDialog),
	layout_(new QVBoxLayout()),
	pushUpSpacer_(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding)),
	palette_(palette){

	ui_->setupUi(this);

	layout_->setContentsMargins(10,10,10,10);
	ui_->scrollArea->setLayout(layout_);
}

TFCompositionDialog::~TFCompositionDialog(){}

bool TFCompositionDialog::refreshSelection(){

	Common::TimeStamp stamp = palette_->lastPaletteChange();
	if(lastPaletteChange_ != stamp)
	{
		lastPaletteChange_ = stamp;

		clearLayout_();
	
		std::set<TF::Size> indexes;
		QCheckBox* editorCheck;
		bool wrongDimension;
		bool isComposition;
		Composition allEditors = palette_->getEditors();
		for(Composition::iterator it = allEditors.begin(); it != allEditors.end(); ++it)
		{
			isComposition = (*it)->hasAttribute(TFBasicHolder::Composition);
			wrongDimension = ((*it)->getDimension() != TF_DIMENSION_1);
			if(isComposition || wrongDimension) continue;

			editorCheck = new QCheckBox(QString::fromStdString((*it)->getName()));
			if(indexesMemory_.find((*it)->getIndex()) != indexesMemory_.end())
			{
				editorCheck->setChecked(true);
				indexes.insert((*it)->getIndex());
			}

			checkBoxes_.push_back(editorCheck);
			layout_->addWidget(editorCheck);

			allAvailableEditors_.push_back(*it);
		}
		layout_->addItem(pushUpSpacer_);
		indexesMemory_.swap(indexes);
	}
	return !allAvailableEditors_.empty();
}

void TFCompositionDialog::accept(){

	indexesMemory_.clear();
	for(TF::Size i = 0;	i < checkBoxes_.size();	++i)
	{
		if(checkBoxes_[i]->isChecked()) 
		{
			indexesMemory_.insert(allAvailableEditors_[i]->getIndex());
		}
	}
	QDialog::accept();
}

void TFCompositionDialog::reject(){

	bool wasChecked;
	for(TF::Size i = 0;	i < checkBoxes_.size();	++i)
	{
		wasChecked = (indexesMemory_.find(allAvailableEditors_[i]->getIndex()) != indexesMemory_.end());
		checkBoxes_[i]->setChecked(wasChecked);
	}
	QDialog::reject();
}

TFCompositionDialog::Composition TFCompositionDialog::getComposition(){

	Composition result;
	for(TF::Size i = 0;	i < checkBoxes_.size();	++i)
	{
		if(checkBoxes_[i]->isChecked()) 
		{
			result.push_back(allAvailableEditors_[i]);
		}
	}
	return result;
}

void TFCompositionDialog::clearLayout_(){

	layout_->removeItem(pushUpSpacer_);
	QLayoutItem* layoutIt;
	while(!layout_->isEmpty())
	{
		layoutIt = layout_->itemAt(0);
		layout_->removeItem(layoutIt);
		layoutIt->widget()->hide();
		delete layoutIt;
	}
	checkBoxes_.clear();
	allAvailableEditors_.clear();
}

} // namespace GUI
} // namespace M4D