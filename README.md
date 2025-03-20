# MLOps Project Setup

This repository contains the setup for the MLOps project as part of the course. The goal is to set up a virtual environment, install necessary dependencies, and prepare the project for development.
All the course labs, assignments will be stored as part of this repository.

---


## 🛠️ Setup Instructions

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd mlops
```

### Step 2: Create a Virtual Environment
- Using venv:
```bash
python -m venv mlops_env
# Activate
source mlops_env/bin/activate
```

Using conda:
```
conda create --name mlops python=3.x
conda activate mlops
```

### Step 3: Install dependencies

```
pip install -r requirements.txt
```

### Step 4: Check installed packages:

```
pip freeze
```
