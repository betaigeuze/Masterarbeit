FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/betaigeuze/Masterarbeit.git .

RUN apt-get update

RUN apt-get install -y graphviz graphviz-dev

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

ENTRYPOINT ["streamlit", "run", "/app/src/dashboardv1/st_dashboard.py", "--server.port=8501", "--server.address=http://172.21.102.175"]