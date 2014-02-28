#include "MedV4D/GUI/TF/CompositionDialog.h"

#include "MedV4D/GUI/TF/Palette.h"
#include "MedV4D/GUI/TF/Editor.h"

namespace M4D {
namespace GUI {

CompositionDialog::CompositionDialog(QWidget* parent):
	QDialog(parent),
	previewEnabled_(false),
	ui_(new Ui::CompositionDialog),
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

CompositionDialog::~CompositionDialog(){

	delete ui_;
}

void CompositionDialog::updateSelection(const std::map<TF::Size, Editor*>& editors, Palette* palette){

	clearLayout_();

	Buttons newButtons;

	Buttons::iterator found;
	PaletteCheckButton* button;
	bool available;
	TF::Size rowCounter = 0, colCounter = 0;
	for(Palette::Editors::const_iterator it = editors.begin(); it != editors.end(); ++it)
	{
		available = !it->second->hasAttribute(Editor::Composition);
		available = available && (it->second->getDimension() == TF_DIMENSION_1);

		found = buttons_.find(it->second->getIndex());
		if(found == buttons_.end())
		{
			button = new PaletteCheckButton(it->second->getIndex());
			button->setup(it->second->getName(), previewEnabled_);
			button->setPreview(palette->getPreview(it->second->getIndex()));
			button->setAvailable(available);
			button->setActive(indexes_.find(it->second->getIndex()) != indexes_.end());

			bool buttonConnected = QObject::connect(button, SIGNAL(Triggered(TF::Size)), this, SLOT(button_triggered(TF::Size)));
			tfAssert(buttonConnected);

			newButtons.insert(std::pair<TF::Size, PaletteCheckButton*>(it->second->getIndex(), button));
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

void CompositionDialog::button_triggered(TF::Size index){

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

void CompositionDialog::accept(){

	indexesMemory_ = indexes_;
	QDialog::accept();
}

void CompositionDialog::reject(){

	indexes_ = indexesMemory_;
	QDialog::reject();
}

CompositionDialog::Selection CompositionDialog::getComposition(){

	selectionChanged_ = false;
	return indexes_;
}

bool CompositionDialog::selectionChanged(){

	return selectionChanged_;
}

void CompositionDialog::clearLayout_(){

	QLayoutItem* layoutIt;
	while(!layout_->isEmpty())
	{
		layoutIt = layout_->itemAt(0);
		layout_->removeItem(layoutIt);
		layoutIt->widget()->hide();
	}
}

void CompositionDialog::resizeEvent(QResizeEvent*){

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

void CompositionDialog::on_previewsCheck_toggled(bool enable){

	previewEnabled_ = enable;
	for(Buttons::iterator it = buttons_.begin(); it != buttons_.end(); ++it)
	{
		it->second->togglePreview(previewEnabled_);
	}
}

} // namespace GUI
} // namespace M4D
