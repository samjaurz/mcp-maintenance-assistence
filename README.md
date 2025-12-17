docker folder run 
docker-compose -p inventory up -d


uv sync
uvicorn main:app --reload

migrations commands
alembic revision --autogenerate -m "init"
alembic upgrade heads
alembic downgrade -1
alembic current
alembic history