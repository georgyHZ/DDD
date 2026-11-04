FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic
COPY api ./api
COPY output/api/metadata.db ./output/api/metadata.db
ENV METADATA_DB=/app/output/api/metadata.db
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
