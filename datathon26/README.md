# Datathon 2026: RadHarmony

Follow these steps to get set up before the workshop.

1. Visit the project on GitHub: https://github.com/f10409/RadHarmony

2. Open a terminal and go to your home directory:

   ```bash
   cd ~
   ```

3. Clone the repository:

   ```bash
   git clone https://github.com/f10409/RadHarmony.git
   ```

4. Move into the datathon folder and run the environment installer. This builds
   the three model environments (`.venv-medgemma`, `.venv-chexagent`,
   `.venv-maira2`) and registers a Jupyter kernel for each. It takes several
   minutes and downloads a lot of packages, so start it first and let it run.

   ```bash
   cd RadHarmony/datathon26
   ./radharmony_venv_installation.sh
   ```

5. Open [0_setup.ipynb](0_setup.ipynb) to log in to Hugging Face and finish the
   setup.
