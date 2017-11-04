FROM tensorflow/tensorflow:1.1.0

RUN pip install yahoo-finance &&\
    groupadd tfgroup &&\
    useradd -g tfgroup --create-home --shell /bin/bash tfuser

USER tfuser

WORKDIR "/home/tfuser"

CMD ["/bin/bash"]
