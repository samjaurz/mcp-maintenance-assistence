docker folder run 
docker-compose -p inventory up -d


uv sync


migrations commands
alembic revision --autogenerate -m "init"
alembic upgrade heads
alembic downgrade -1
alembic current
alembic history


Install git
Clone de repository
Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
