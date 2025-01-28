FROM python:3.11.11-bookworm@sha256:adb581d8ed80edd03efd4dcad66db115b9ce8de8522b01720b9f3e6146f0884c

RUN pip install autogluon==1.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

RUN mkdir training_data
COPY ./data/NED training_data
COPY ./train_model.py train_model.py
COPY ./read_ned.py read_ned.py

ENV training_data=training_data
ENV model_path=model

RUN python train_model.py

RUN rm training_data -rf

COPY ./retrieve_ned.py retrieve_ned.py
COPY ./predict.py predict.py

CMD [ "python", "predict.py" ]
