set -e
 

module --force purge 
module load env/release/2024.1 
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

python3 -m mlflow server   --host localhost  --port 5000   --backend-store-uri file:$(pwd)/_saved/mlflow   --default-artifact-root  file:$(pwd)/_saved/mlflow
 