#include "FranghisVesselnessOptions.h"

FranghisVesselnessOptions::FranghisVesselnessOptions() : dialog(new Ui::FranghisVesselnessOptions())
{
  this->dialog->setupUi(this);
}

FranghisVesselnessOptions::~FranghisVesselnessOptions()
{
  delete this->dialog;
}