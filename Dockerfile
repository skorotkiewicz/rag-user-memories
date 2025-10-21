FROM postgres:14

ENV POSTGRES_DB=vectordb
ENV POSTGRES_USER=vectoruser
ENV POSTGRES_PASSWORD=vectorpass

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-14

RUN git clone https://github.com/pgvector/pgvector.git

RUN cd pgvector && \
    make && \
    make install

RUN apt-get remove -y build-essential git postgresql-server-dev-14 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /pgvector

# RUN echo "CREATE EXTENSION vector;" > /docker-entrypoint-initdb.d/init.sql
