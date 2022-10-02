# Petals: Decentralized platform for running 100B+ language models

## Key features

- Run inference or fine-tune large language models like [BLOOM-176B](https://huggingface.co/bigscience/bloom) by joining compute resources with people all over the Internet. No need to have high-end GPUs.
- It's difficult to fit the whole BLOOM-176B into GPU memory unless you have multiple high-end GPUs. Instead, **Petals** allows to load and serve a small part of the model, then team up with people serving all the other parts to run inference or fine-tuning.
- This way, one inference step takes ≈ 1 sec — much faster than possible with offloading. Enough for chatbots and other interactive apps.
- Beyond traditional language model APIs — you can employ any fine-tuning and sampling methods by executing custom paths through the model or accessing its hidden states. This allows for the comforts of an API with the flexibility of PyTorch.

## How it works?

<p align="center">
    <img src="https://i.imgur.com/RTYF3yW.png" width="800">
</p>

### 🛠️ Examples

Petals integrates seamlessly with PyTorch and the Hugging Face [Transformers](https://github.com/huggingface/transformers) library.

This snippet shows how to **(a)** generate text with BLOOM and **(b)** solve a sequence classification task via soft prompt tuning:

```python
# Initialize distributed BLOOM and connect to the swarm
model = DistributedBloomForCausalLM.from_pretrained(
    PRETRAINED_PATH, tuning_mode="ptune", initial_peers=SEE_BELOW
)  # Embeddings & prompts are on your device, BLOOM blocks are distributed

print("Generated:", model.generate(tokenized_prefix, max_new_tokens=5))

# Training (updates only local prompts / adapters)
optimizer = torch.optim.AdamW(model.parameters())
for input_ids, labels in data_loader:
    outputs = model.forward(input_ids)
    loss = cross_entropy(outputs.logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 🔒 Privacy and security

If you work with sensitive data, you should only use a private swarm (or a subset of servers in the public swarm) hosted by people and institutions you trust, who are authorized to process this data.

This is important because it's technically possible for peers serving model layers to recover input data or model outputs. Also, if there are malicious peers, they may alter their outputs to influence the model outputs. See a more detailed discussion in our paper.

## Installation

Here's how to install the dependencies with conda:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

This script uses Anaconda to install cuda-enabled PyTorch.
If you don't have anaconda, you can get it from [here](https://www.anaconda.com/products/distribution).
If you don't want anaconda, you can install PyTorch [any other way](https://pytorch.org/get-started/locally/).
If you want to run models with 8-bit weights, please install **PyTorch with CUDA 11** or newer for compatility with [bitsandbytes](https://github.com/timDettmers/bitsandbytes).

__OS support:__ Currently, Petals only supports Linux operating systems. On Windows 11, you can run Petals with GPU enabled inside WSL2 ([read more](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)).
For macOS, you can *probably* run everything normally if you manage to install dependencies, but we do not guarantee this.


## 🚀 Getting Started

This is a toy example running on a local machine without GPU and with a tiny model.
For a detailed instruction with larger models, see ["Launch your own swarm"](./docs/Launch-your-own-swarm.md).

First, run a couple of servers, each in a separate shell. To launch your first server, run:
```bash
python -m cli.run_server bloom-testing/test-bloomd-560m-main --num_blocks 8 --torch_dtype float32 \
  --host_maddrs /ip4/127.0.0.1/tcp/31337   # use port 31337, local connections only
```

This server will host 8 (out of 24) blocks of a [tiny 560M version](https://huggingface.co/bloom-testing/test-bloomd-560m-main) of the BLOOM model that was converted for Petals.

> If you'd like to run a swarm of servers with the full BLOOM straight away, please see [this instruction](./docs/Launch-your-own-swarm.md) (you'll need several GPUs!). To run a different model, see [this tutorial](./docs/Run-a-custom-model-with-Petals.md).

Once the server has started, it will print out a ton of information, including an important line like this:

```bash
Mon Day 01:23:45.678 [INFO] Running DHT node on ['/ip4/127.0.0.1/tcp/31337/p2p/ALongStringOfCharacters'], initial peers = []
```

You can use this address (`/ip4/whatever/else`) to connect additional servers. Open another terminal and run:

```bash
python -m cli.run_server bloom-testing/test-bloomd-560m-main --num_blocks 8 --torch_dtype float32 \
  --host_maddrs /ip4/127.0.0.1/tcp/0 \
  --initial_peers /ip4/127.0... # <-- TODO: Copy the address of another server here
# e.g. --initial_peers /ip4/127.0.0.1/tcp/31337/p2p/QmS1GecIfYouAreReadingThisYouNeedToCopyYourServerAddressCBBq
```

You can assign `--initial_peers` to one or multiple addresses of other servers, not necessarily the first one.
The only requirement is that at least one of them is running at the time.

Before you proceed, __please run 3 servers__ for a total of 24 blocks (3x8). If you are running a different model,
make sure your servers have enough total `--num_blocks` to cover that model. 

Once your have enough servers, you can use them to train and/or inference the model:
```python
import torch
import torch.nn.functional as F
import transformers
from src import DistributedBloomForCausalLM

initial_peers = [TODO_put_one_or_more_server_addresses_here]  # e.g. ["/ip4/127.0.0.1/tcp/more/stuff/here"]
tokenizer = transformers.BloomTokenizerFast.from_pretrained("bloom-testing/test-bloomd-560m-main")
model = DistributedBloomForCausalLM.from_pretrained(
  "bloom-testing/test-bloomd-560m-main", initial_peers=initial_peers, low_cpu_mem_usage=True, torch_dtype=torch.float32
)  # this model has only embeddings / logits, all transformer blocks rely on remote servers


inputs = tokenizer("a cat sat", return_tensors="pt")["input_ids"]
remote_outputs = model.generate(inputs, max_length=10)
print(tokenizer.decode(remote_outputs[0]))  # "a cat sat in the back of the car,"

# "train" input embeddings by backprop through distributed transformer blocks
model.transformer.word_embeddings.weight.requires_grad = True
outputs = model.forward(input_ids=inputs)
loss = F.cross_entropy(outputs.logits.flatten(0, 1), inputs.flatten())
loss.backward()
print("Gradients (norm):", model.transformer.word_embeddings.weight.grad.norm())
```

Of course, this is a simplified code snippet. For actual training, see our example on "deep" prompt-tuning here: [examples/prompt-tuning-personachat.ipynb](./examples/prompt-tuning-personachat.ipynb).

Here's a [more advanced tutorial](./docs/Launch-your-own-swarm.md) that covers 8-bit quantization and best practices for running Petals.