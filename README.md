# Forecasting grid emission factor for the Netherlands

This repo contains a workflow to produce emission factor forecasts for the Dutch
electricity mix, 7 days ahead.

## Model training
Model training is performed with [AutoGluon](https://auto.gluon.ai/), using the time
series forcasting module.

As training data the total energy production and the energy mix's emission factor are
used, sourced from the [Nationaal Energie Dashboard](https://ned.nl/)
with the produced solar and wind energy as "known covariates".

<img src="model_test.png" alt="Model training test result" width="400"/>

*Model training result, validation on unseen data*

The NED also provides forecasts for the solar and wind production.
These are used in forecasting of the emission factor.

<img src="example_forecast.png" alt="Example forecast" width="400"/>

*Example forecast, for 2025-01-28 - 2025-02-04*

## Reproducing results

The Dockerfile contained in this reposity describes all steps you need to go
through to train a model and to produce a forecast.

To run the docker file do:

```docker
docker run -e ned_api_key -e csv_path=/data/prediction.csv --volume /data:/data ghcr.io/bschilperoort/emissionfactor-forecast
```

Where the environmental variable `ned_api_key` should be your ned.nl API key,
and the `/data` directory the location where the prediction file should end up.

Note that for model training, historical NED data is required, but this is removed
from the Docker image due to licensing restrictions.

## ToDo:
This is a work in progress, the repository will likely be reorganized and more
descriptive documentation needs to be added.
