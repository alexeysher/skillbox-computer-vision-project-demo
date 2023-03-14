# import argparse

import streamlit.web.bootstrap as bootstrap

# parser = argparse.ArgumentParser()
# args = parser.parse_args()
real_script = 'main.py'

bootstrap.run(real_script, f'run.py {real_script}', [], {})
