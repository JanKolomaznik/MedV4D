class myCallback : public vtkCommand
{
public:
  vtkImageViewer *IV;
  vtkImageIdealLowPass *i2;

  static myCallback *New() { return new myCallback; }

  void Execute(vtkObject *caller,
               unsigned long event,
               void *vtkNotUsed(callData))
    {
      if (this->IV->GetInput() == NULL) { return; }

      // Reset

      if (event == vtkCommand::ResetWindowLevelEvent)
        {
        this->IV->GetInput()->UpdateInformation();
        this->IV->GetInput()->SetUpdateExtent(this->IV->GetInput()->GetWholeExtent());
        this->IV->GetInput()->Update();

        this->i2->SetXCutOff(0);
        this->i2->SetYCutOff(0);

		this->IV->Render();
        return;
        }

      // Start

      if (event == vtkCommand::StartWindowLevelEvent)
        {
          return;
        }

      // Adjust the window level here

      vtkInteractorStyleImage *isi = static_cast<vtkInteractorStyleImage *>(caller);

      int *size = this->IV->GetRenderWindow()->GetSize();

      // Compute normalized delta

      double dx = 1.0 * (isi->GetWindowLevelCurrentPosition()[0] - isi->GetWindowLevelStartPosition()[0]) / size[0];
      double dy = 1.0 * (isi->GetWindowLevelStartPosition()[1] - isi->GetWindowLevelCurrentPosition()[1]) / size[1];

      // Scale by current values

      this->i2->SetXCutOff(fabs(dx));
      this->i2->SetYCutOff(fabs(dy));

      this->IV->Render();
    }
};

