FROM amazonlinux

# Install python for running the server and net-tools for modifying network config
RUN yum install python3 net-tools python-pip -y

WORKDIR /app

COPY requirements.txt ./
RUN python3 -m pip --version
RUN python3 -m pip install -r /app/requirements.txt

COPY server.py ./
COPY server.py ./server.py2
COPY NsmUtil.py ./
COPY run.sh ./
COPY libnsm.so ./

RUN chmod +x run.sh

CMD ["/app/run.sh"]
