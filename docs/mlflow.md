# MLFlow 
if activated one have to do this before the run
```bash
python3 -m mlflow server \
  --host localhost\
  --port 5000 \
  --backend-store-uri file:path/to/the/directory
```