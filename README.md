# Alcea

**LLM-based Translation System**
A modular pipeline for NLP, translation, and speaker-aware literary processing using Conda and Flask.

---

## 🔧 Setup Instructions

### 1. Download and Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda
```

Installs Miniconda locally in `./miniconda`.

---

### 2. Create Conda Environments

Create a folder for the environments:

```bash
mkdir -p envs
```

Set up specific environments:

* **LLM Environment (Python 3.12)**:

```bash
conda create --prefix ./envs/llm_env python=3.12 -y
```

* **BookNLP Environment (Python 3.9)**:

```bash
conda create --prefix ./envs/booknlp_env python=3.9 -y
```

* **Flask Environment (Python 3.12)**:

```bash
conda create --prefix ./envs/flask_env python=3.12 -y
```

---

### 3. Environment Activation

Make activation script executable:

```bash
chmod +x env.sh
```

Activate the desired environment:

```bash
source env.sh [book|llm|flask]
```

Example:

```bash
source env.sh book
```

---

### 4. Environment-Specific Setup

#### 📚 BookNLP

```bash
source env.sh book
python -m spacy download en_core_web_sm
pip install -r nlp_requirements.txt
```

#### 🤖 LLM

```bash
source env.sh llm
pip install -r llm_requirements.txt
```
> Note: To work with models with a large number of parameters, it is necessary to set the secret key for inference = InferenceClient
#### 🌐 Flask

```bash
source env.sh flask
pip install -r interface/flask_requirements.txt
```

---

### 5. Experiment Visualization (Optional)

Set up Python virtual environment:

```bash
python3 -m venv experiments_evals/venv
source experiments_evals/venv/bin/activate
pip install -r experiments_evals/requirements.txt
```

---

## 🚀 Running Flask Server

Activate Flask environment:

```bash
source env.sh flask
```

Launch the server:

```bash
python interface/app.py
```

---

## 📁 Repository Structure

```
.
├── env.sh                      # Activation script
├── envs/                       # Conda environments
├── interface/
│   ├── app.py                  # Flask server
│   └── flask_requirements.txt
├── experiments_evals/
│   └── requirements.txt        # Visualization requirements
├── llm_requirements.txt
├── nlp_requirements.txt
├── miniconda/                  # Local Miniconda installation
└── ...
```

---

## ✨ License

This project is intended for academic and personal research purposes.
