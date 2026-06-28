# On Miluxina

## Interactive session 
```bash
# Development
salloc -A p200177 -t 4:00:00 -q dev --reservation=cpudev -p cpu -N 1
salloc -A p200177 -t 4:00:00 -q dev --reservation=gpudev -p gpu  -N 1
# Production 
salloc -A p200177  -p gpu --qos default -N 1 -t 12:00:00
```


## Python
```bash
- env 
#Create the environment
module --force purge 
module load env/release/2024.1 
module load Python/3.12.3-GCCcore-13.3.0
python -m venv .venv 
source .venv/bin/activate
pip3 install --upgrade pip
pip install -e . 
```
- test and dev 
```bash
pip install -e ".[test]"
pip install -e ".[dev]"
```
## NetCDF
```
module load env/release/2025.1
mole load netCDF-Fortran/4.6.2-iimpi-2025a
```

