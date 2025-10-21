docker build -t postgres-pgvector .

docker run -d --name postgres-vector -p 5432:5432 postgres-pgvector

docker stop postgres-vector
docker start postgres-vector
docker restart postgres-vector

pip install psycopg2-binary pgvector sentence-transformers torch transformers

