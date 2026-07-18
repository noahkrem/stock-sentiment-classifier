# stock-sentiment-classifier

## Setup

### Creating a Virtual Environment

To create a Python virtual environment for this project, run:

```bash
python3 -m venv venv
```

This creates a `venv` directory containing the isolated Python environment.

### Activating the Virtual Environment

Once created, activate the virtual environment:

**On Linux/macOS/WSL:**
```bash
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

When activated, you should see `(venv)` prefix in your terminal prompt.

### Installing Requirements

After activating the virtual environment, install all required packages:

```bash
pip install -r requirements.txt
```

### Deactivating the Virtual Environment

When finished, deactivate the virtual environment by running:

```bash
deactivate
```

## Running the Labeling Pipeline

To generate the labeled weekly dataset from the raw input files, run:

```bash
python3 src/label_data.py
```

This creates the output file at data/processed/labeled.csv.

## Training the model

To train the model on your own machine navigate to the root directory for the project and run:

```bash
python3 src/train.py --data data/processed/labeled.csv --out models/model.pkl
```

This will run train.py using the data in labeled.csv and output the resulting model in the 'models' folder.
