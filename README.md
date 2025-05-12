#Setup
Create a conda environment with python 3.12


```bash
python3 -m pip install -r requirements.txt

#GPT-2 Chatbot Acceleration Benchmark

This project benchmarks multiple GPU/CPU training acceleration techniques for a GPT-2 based chatbot using the [DailyDialog](https://huggingface.co/datasets/daily_dialog) dataset.

Goal:

Compare training performance, memory usage, and token-level accuracy across:

-  CPU (multi-threaded & multi-process)
- DDP (DistributedDataParallel), Accelerate (HuggingFace), DeepSpeed (Microsoft)

Each method uses the same tokenizer, dataset, model structure (GPT-2), and evaluation logic to ensure fair comparison.
 Project Structure

