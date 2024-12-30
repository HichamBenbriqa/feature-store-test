FROM python:3.9-slim

WORKDIR /app

RUN pip install poetry==1.8.2

COPY pyproject.toml poetry.lock ./

COPY . .

RUN poetry install --no-root --only main

CMD ["poetry", "run", "python", "main.py"]