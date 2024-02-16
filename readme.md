# Enhancing Wound Healing Using Deep Reinforcement Learning

## Open mapperPaperResGen.ipynb for figures in the paper

## To run locally

1. Clone the repository to your local computer
2. Create a virtual python environment: python -m venv /path/to/new/virtual/environment
3. Activate your virtual env: if you use Linux/Mac: venv/bin/activate; if you use Windows: venv/Scripts/activate
4. Install all the required packages: pip install -r requirements.txt
5. In the main file, change the colab_dir to the one your would like to save all your experiments data
6. Run: python main_map54_v2.py
7. Open another terminal, cd to your data saved directory set by colab_dir
8. Run tensorboard --logdir=runs_map524 --port=6006
9. In your brower, type: localhost:6006
