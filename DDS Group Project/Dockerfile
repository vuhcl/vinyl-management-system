FROM apache/airflow:3.1.6

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64

USER airflow
RUN pip install --no-cache-dir \
    apache-airflow-providers-google \
    gcsfs \
    pymongo \
    torch \
    pyspark==3.5.5 \
    apache-airflow-providers-apache-spark
