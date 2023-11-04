# Use the Ubuntu base image
FROM ubuntu:22.04

# Update The Package
RUN apt-get update -y 
RUN apt-get install -y

# Install Python 3 and pip
RUN apt-get install -y python3 python3-pip

# Install Python python libraries
RUN pip3 install pandas 
RUN pip3 install numpy
RUN pip3 install seaborn
RUN pip3 install matplotlib
RUN pip3 install scikit-learn
RUN pip3 install scipy

# Move Dataset
RUN mkdir /home/doc-bd-a1/
COPY titanic.csv/ /home/doc-bd-a1/

WORKDIR /home/doc-bd-a1/

# Move the .py files
COPY load.py /home/doc-bd-a1/
COPY dpre.py /home/doc-bd-a1/
COPY eda.py /home/doc-bd-a1/
COPY vis.py /home/doc-bd-a1/
COPY model.py /home/doc-bd-a1/

# Run the python files
RUN python3 load.py titanic.csv

# Run the container bash
CMD ["bash"]