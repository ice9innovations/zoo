FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

RUN pip install speciesnet==5.0.3 flask==3.1.2 flask-cors==6.0.1 python-dotenv==1.2.1

COPY REST.py /app/
COPY family_groups.csv /app/

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 7794

CMD ["python", "REST.py"]
