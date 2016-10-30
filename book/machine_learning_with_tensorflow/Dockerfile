FROM tensorflow/tensorflow

RUN pip install yahoo-finance &&\
    groupadd tfgroup &&\
    useradd -g tfgroup --create-home --shell /bin/bash tfuser

WORKDIR "/home/tfuser"

CMD ["/bin/bash"]
