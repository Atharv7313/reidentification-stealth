# Clone the repo
git clone https://github.com/Atharv7313/reidentification-stealth.git
cd reidentification-stealth

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training or inference
python train.py  # or whichever script starts the pipeline

## Make sure best.pt is present in the src folder before executing the file.
