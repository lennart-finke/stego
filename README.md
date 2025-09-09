# Breaking Down Language Model Steganography with External Entropy 📨

In steganography, two actors conspire with each other to transmit a message over a surveilled channel, in our case through text.

We focus on the setting where the sender and receiver of the secret message are large language models: Can they exploit external entropy to encode the message? Can they use their own parameters as a source of entropy?

## Paper



## Usage
For setting up model model provider keys, create an `.env` from `.env.example`.

This repository uses [Inspect](https://inspect.ai-safety-institute.org.uk). After installing it, you can for instance run `inspect eval steganography.py --model anthropic/claude-3-5-haiku-latest --limit 2` and see the results with `inspect view`.

For running the eval on a combination of encoder and decoder models, we can use `eval.sh` (which depends on `tmux`) and analyze the results with `analyze_logs.py`.

## Reproduction
### 4.1 Basic Encryption
- For **External Key**, run `eval.sh` with `o3-mini`, `deepseek-r1` and `gemini-2.0-flash-thinking` as `encoder_models` and `decoder_models`, with `DATASET=sample.jsonl` in `steganography.py`. Then, run `analyze_logs.json` with the same choice of `encoder_models` and `decoder_models`.
- For **Payload Length Ablation**, `inspect eval steganography.py --model openai/o3-mini --epochs 2 -T "decoder_model=openai/o3-mini" --log-format json`, then `analyze_logs_varied_length.py`, with an adjusted `log_path` if applicable.
- For **Natural Language Schelling Points**, run `get_noun_answers.py`, then `analyze_nouns.py`. For the finetuning experiment, generate the dataset with `prepare_finetuning.py`. The OpenAI website was used to launch the finetuning job, monitoring and obtaining the model id can be done in `get_finetuned_model.ipynb`.
- For **Internal Key**, run `eval.sh` with `o3-mini`, `deepseek-r1` and `gemini-2.0-flash-thinking` as `encoder_models` and `decoder_models`, with `DATASET=self-entropy.jsonl` in `steganography.py`. Then, run `analyze_logs.json` with the same choice of `encoder_models` and `decoder_models`. For the bottom of Girue 3, the same can be done again for the finetuned model by entering the model id from `get_finetuned_model.ipynb` as `encoder_models` as well as the `decoder_model`.

### 4.2 End-to-End Steganography
Run `inspect eval steganography_end2end.py --model openai/o3-mini  -T "decoder_model=openai/o3-mini" --log-format json`, then `analyze_steganography_end2end.py`. This produces Figure 2 and 7.

## Acknowledgments
Many thanks to Christian Schroeder de Witt for advice and feedback.