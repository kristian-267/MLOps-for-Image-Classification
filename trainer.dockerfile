# Base image
FROM python:3.10

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_tests.txt requirements_tests.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY tests/ tests/
COPY conf/ conf/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_tests.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
