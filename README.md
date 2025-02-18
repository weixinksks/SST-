# SST-
A Transformer-based Method for Correcting Daily SST Numerical Forecasting products

## Project Description:

This project presents a Transformer-based approach for correcting daily sea surface temperature (SST) numerical forecast products. Traditional SST forecasting methods often suffer from model errors and limited observational data. By leveraging the advanced Transformer architecture and integrating spatio-temporal features, this approach aims to enhance the accuracy and reliability of numerical forecasts. The method holds significant potential for applications in marine environment monitoring, climate prediction, and navigation safety.

## Project structure:

```yaml
/code                          # Code section
|-- Geoformer.py                # Implementation of the Transformer model
|-- SST.py                       # Data preprocessing and augmentation code, including data loading, normalization, etc.
|-- my_tools.py                 # Utility functions, including details of the Transformer structure implementation
|-- myconfig.py                 # Configuration file, including training parameters, hyperparameters, etc.
|-- trainer_2.py                # Main entry point for model training

/data                          # Data section
|-- SST.nc                      # Raw SST dataset, including ERA5 reanalysis data, model data, etc.
|-- err.nc                      # Processed error dataset

```

## Additional notes：

Due to the storage limitations of the website, our original dataset is uploaded on Google Drive. The link is: https://drive.google.com/file/d/1X8z-_1N4pU0YM_XMATPWHaqxc38ENmds/view?usp=drive_link



