# agent-woko
Research and labs around agentic workflows and development
### `uv`

Follow the instructions here to install uv

https://docs.astral.sh/uv/getting-started/installation/

Start by running `uv self update` to make sure you're on the latest version of uv.

And now simply run:  
`uv sync`  
If necessary, uv should install python 3.12, and then it should install all the packages.  
If you get an error about "invalid certificate" while running `uv sync`, then try this instead:  
`uv --native-tls sync`  
And also try this instead:  
`uv --allow-insecure-host github.com sync`

Finally, run these commands to be ready to use CrewAI:  
`uv tool install crewai`   
Followed by:  
`uv tool upgrade crewai`  

Checking that everything is set up nicely:  
1. Confirm that you now have a folder called '.venv' in your project root directory (agents)
2. If you run `uv python list` you should see a Python 3.12 version in your list (there might be several)
3. If you run `uv tool list` you should see crewai as a tool

Just FYI on using uv:  
With uv, you do a few things differently:  
- Instead of `pip install xxx` you do `uv add xxx` - it gets included in your `pyproject.toml` file and will be automatically installed next time you need it  
- Instead of `python my_script.py` you do `uv run my_script.py` which updates and activates the environment and calls your script  
- You don't actually need to run `uv sync` because uv does this for you whenever you call `uv run`  
- It's better not to edit pyproject.toml yourself, and definitely don't edit uv.lock. If you want to upgrade all your packages, run `uv lock --upgrade`
- uv has really terrific docs [here](https://docs.astral.sh/uv/) - well worth a read!