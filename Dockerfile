FROM python:3.9-slim

WORKDIR /app

RUN pip install poetry==1.8.2

COPY pyproject.toml poetry.lock ./

COPY . .

RUN poetry install --no-root --only main

CMD ["poetry", "run", "python", "main.py"]


#  docker build -t pipedrive-test .

# docker run \
#     -e AWS_ACCESS_KEY=AKIATB7GH52TEYSPWHHY \
#     -e AWS_SECRET_ACCESS_KEY=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 \
#     -e AWS_DEFAULT_REGION=us-east-1 \
#     -e AWS_ROLE=arn:aws:iam::210399391398:role/service-role/AmazonSageMaker-ExecutionRole-20241221T151049 \
#     pipedrive-test:latest