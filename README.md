## Setup

Create a conda environment with python 3.9+

```bash
conda create -n mvenv python=3.9
conda activate mvenv
```

Install requirements

```bash
python3 -m pip install -r requirements.txt
```

## Usage

Comparison of different GPU/CPU training acceleration techniques for a GPT-2 based chatbot using the [DailyDialog](https://huggingface.co/datasets/daily_dialog) dataset.

```bash
Calling a Chatbot with
```

Render an interactive visualization of the shortest path between two points

```python
from astar import astar
from visualize import visualize

# load saved map from earlier step
