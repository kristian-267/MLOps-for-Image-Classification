# Base image
FROM python:3.10

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY models/deployable_model.pt models/deployable_model.pt
COPY src/ src/
COPY conf/ conf/

RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install --no-cache-dir --upgrade opencv-python-headless python-multipart
COPY app/ app/

WORKDIR /

CMD ["uvicorn", "app.cloud_deployment:app", "--host", "0.0.0.0", "--port", "80"]
