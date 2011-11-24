#include "GUI/TF/TFCompositionDialog.h"

#include "GUI/TF/TFPalette.h"
#include "GUI/TF/TFEditor.h"

namespace M4D {
namespace GUI {

TFCompositionDialog::TFCompositionDialog(QWidget* parent):
	QDialog(parent),
	previewEnabled_(false),
	ui_(new Ui::TFCompositionDialog),
	layout_(new QGridLayout()),
	colModulator_(1),
	selectionChanged_(false)
{
	ui_->setupUi(this);

	layout_->setContentsMargins(10,10,10,10);
	layout_->setAlignment(Qt::AlignCenter);
	layout_->setSpacing(5);
	ui_->scrollAreaWidget->setLayout(layout_);
}

TFCompositionDialog::~TFCompositionDialog(){

	delete ui_;
}

void TFCompositionDialog::updateSelection(const std::map<TF::Size, TFEditor*>& editors, TFPalette* palette){

	clearLayout_();

	Buttons newButtons;

	Buttons::iterator found;
	TFPaletteCheckButton* button;
	bool available;
	TF::Size rowCounter = 0, colCounter = 0;
	for(TFPalette::Editors::const_iterator it = editors.begin(); it != editors.end(); ++it)
	{
		available = !it->second->hasAttribute(TFEditor::Composition);
		available = available && (it->second->getDimension() == TF_DIMENSION_1);

		found = buttons_.find(it->second->getIndex());
		if(found == buttons_.end())
		{
			button = new TFPaletteCheckButton(it->second->getIndex());
			button->setup(it->second->getName(), previewEnabled_);
			button->setPreview(palette->getPreview(it->second->getIndex()));
			button->setAvailable(available);
			button->setActive(indexes_.find(it->second->getIndex()) != indexes_.end());

			bool buttonConnected = QObject::connect(button, SIGNAL(Triggered(TF::Size)), this, SLOT(button_triggered(TF::Size)));
			tfAssert(buttonConnected);

			newButtons.insert(std::make_pair<TF::Size, TFPaletteCheckButton*>(it->second->getIndex(), button));
		}
		else
		{
			button = found->second;
			button->setName(it->second->getName());
			button->setPreview(palette->getPreview(it->second->getIndex()));
			button->togglePreview(previewEnabled_);
			button->setAvailable(available);
			button->setActive(indexes_.find(it->second->getIndex()) != indexes_.end());

			newButtons.insert(*found);
			buttons_.erase(found);
		}

		layout_->addWidget(button, rowCounter, colCounter, Qt::AlignCenter);
		button->show();
		++colCounter;
		if(colCounter == colModulator_)
		{
			colCounter = 0;
			++rowCounter;
		}
	}
	for(Buttons::iterator it = buttons_.begin(); it != buttons_.end(); ++it)
	{
		delete it->second;
	}
	buttons_.swap(newButtons);
}

void TFCompositionDialog::button_triggered(TF::Size index){

	Buttons::iterator triggered = buttons_.find(index);
	tfAssert(triggered != buttons_.end());

	if(triggered->second->isActive())
	{
		triggered->second->setActive(false);
		indexes_.erase(index);
	}
	else
	{
		triggered->second->setActive(true);
		indexes_.insert(index);
	}
	
	selectionChanged_ = true;
}

void TFCompositionDialog::accept(){

	indexesMemory_ = indexes_;
	QDialog::accept();
}

void TFCompositionDialog::reject(){

	indexes_ = indexesMemory_;
	QDialog::reject();
}

TFCompositionDialog::Selection TFCompositionDialog::getComposition(){

	selectionChanged_ = false;
	return indexes_;
}

bool TFCompositionDialog::selectionChanged(){

	return selectionChanged_;
}

void TFCompositionDialog::clearLayout_(){

	QLayoutItem* layoutIt;
	while(!layout_->isEmpty())
	{
		layoutIt = layout_->itemAt(0);
		layout_->removeItem(layoutIt);
		layoutIt->widget()->hide();
	}
}

void TFCompositionDialog::resizeEvent(QResizeEvent*){

	ui_->dialogWidget->resize(size());

	TF::Size newColModulator = (ui_->scrollArea->width() - 25)/(buttons_.begin()->second->width() + 5);
	if(newColModulator == 0) newColModulator = 1;

	if(colModulator_ == newColModulator) return;

	colModulator_ = newColModulator;

	clearLayout_();

	TF::Size rowCounter = 0, colCounter = 0;
	for(Buttons::iterator it = buttons_.begin(); it != buttons_.end(); ++it)
	{
		layout_->addWidget(it->second, rowCounter, colCounter, Qt::AlignCenter);
		it->second->show();
		++colCounter;
		if(colCounter == colModulator_)
		{
			colCounter = 0;
			++rowCounter;
		}
	}
}

void TFCompositionDialog::on_previewsCheck_toggled(bool enable){

	previewEnabled_ = enable;
	for(Buttons::iterator it = buttons_.begin(); it != buttons_.end(); ++it)
	{
		it->second->togglePreview(previewEnabled_);
	}
}

} // namespace GUI
} // namespace M4D
