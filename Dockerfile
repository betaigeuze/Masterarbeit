FROM python:3.9-slim

EXPOSE 8501

WORKDIR /Masterarbeit

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y install git

RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN --mount=type=ssh git clone git@github.com:betaigeuze/Masterarbeit.git Masterarbeit

RUN apt-get update

RUN apt-get install -y graphviz graphviz-dev

COPY .streamlit/config.toml /root/.streamlit/config.toml

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

ENTRYPOINT ["streamlit", "run", "Masterarbeit/src/dashboardv1/st_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]