This tutorial will walk you through the steps of setting up your own private swarm to inference and fine-tune BLOOM. Please make sure you have already installed Petals and followed the basic example from the README section.

This tutorial covers BLOOM-176B. It requires ~200GB of combined GPU memory in 8 bit. If you want to try this on a smaller scale, use `bigscience/test-bloomd-6b3` model.


### Step 1: Set up the network

If you plan to work with unreliable GPU servers (e.g. spot instances), it is a good practice to have a few non-GPU devices that are always online. These "backbone" peers can be used as `--initial_peers`, to connect new GPU servers to the existing ones. They can also [serve as relays](https://docs.libp2p.io/concepts/circuit-relay/) for GPU servers that lack open ports.

If you have reliable GPU servers, you can skip this step entirely and use these servers as initial peers, like in the basic tutorial.

To start a non-GPU peer, run this line in a tmux / screen shell:
`hivemind-dht --identity peer1.id --host_maddrs /ip4/0.0.0.0/tcp/8989`

Once you run it, look at the outputs and find the following line:
```
Mon 00 01:23:45.678 [INFO] Running a DHT instance. To connect other peers to this one, use --initial_peers /ip4/YOUR_ADDRESS_HERE/tcp/8989/p2p/QmTPAIfThisIsMyAddressGoFindYoursnCfj
```

You can provide this address as `--initial_peers` to GPU servers or other backbone peers. If there is a risk that this peer goes down, you can launch additional hivemind-dht instances and provide multiple addresses. New peers will be able to join the swarm as long as at least one of their initial peers is alive.

Here's a few tips to help you set up:

__The host_maddrs__ contains "multi-addresses" containing an IP address, port and network protocols. Learn more about them in [this guide](https://docs.libp2p.io/concepts/addressing/).
   * The last part of a multi-address defines the network port (`8989`), which should be accessible to other peers. You can set port to 0 to choose it at random.
   * Depending on your network, you may need to manually dial your IP to avoid connection issues, e.g. `/ip4/12.34.56.78/tcp/8989`
     When running over the internet, you can auto-detect IP with this script:
```bash
        export IPV4=$(dig -4 TXT +short o-o.myaddr.l.google.com @ns1.google.com |  tr -d '"')
        export IPV6=$(dig -6 TXT +short o-o.myaddr.l.google.com @ns1.google.com |  tr -d '"')
        echo "My IP v4: [ $IPV4 ] v6: [ $IPV6 ] - must be non-empty!"  # if IP is empty, the script has failed (e.g. no internet)
```

__The identity__ defines the "p2p/QmWhatever" part of your peer's address. __Each peer's identity must be unique!__
   * set `--identity` option to a file (created if missing) to ensure that your peer has the same identity each time you restart it;
   * if you omit this option, Petals will generate a new identity each time a process is started. This is fine for "temporary" peers.
   


### Step 2: Start Petals servers

We will run [`bloom-petals`](https://huggingface.co/bigscience/bloom-petals) - the BLOOM-176B model that was [converted to Petals format](../blob/main/docs/Run-a-custom-model-with-PETALS.md).

Here's the full script that we used to benchmark Petals over the internet.
Don't worry, we'll explain everything.

```bash
export CUDA_VISIBLE_DEVICES="0" # choose one GPU index (e.g. "0") or leave blank to run on CPU
export PETALS_8BIT_BACKWARD=1  # enable backward pass when in load_in_8bit (r/n this causes a slightly slower forward)
export NUM_BLOCKS=<TODO pick the number of blocks based on your GPU memory, see below>
export COMPRESSION=<use "NONE" when running locally or "BLOCKWISE_8BIT" to run over the internet>
export INITIAL_PEERS=<TODO add one or several multi-addresses to connect to>
export CACHE_SIZE="1.0 GiB"  # rule of thumb: 250MB per block, see notes

export PORT=6789   # select an open port
export IPV4=$(dig -4 TXT +short o-o.myaddr.l.google.com @ns1.google.com |  tr -d '"')
echo "My IP v4: [ $IPV4 ] - must be non-empty!"  # if IP is empty, you need to specify IP manually

python -m cli.run_server bigscience/bloom-petals --num_blocks $NUM_BLOCKS --throughput auto \
 --torch_dtype float16 --load_in_8bit --compression $COMPRESSION --attn_cache_size $CACHE_SIZE \
 --host_maddrs /ip4/0.0.0.0/tcp/$PORT /ip4/::/udp/$PORT/quic --announce_maddrs /ip4/$IPV4/tcp/$PORT /ip4/$IPV4/udp/$PORT/quic \
 --identity_path ./agirlhasnoname.id  --initial_peers $INITIAL_PEERS
```

That's a lot of stuff. Let's cover it one parameter at a time:

- __`num_blocks`__ depends on your GPU memory. A good rune of thumb is `num_blocks = (gpu_memory_gb - 2) / 2.75`.
- __`throughput`__ measures your server's throughput for load-balancing. Currently, it runs [speedtest](https://www.speedtest.net/). If it does not work, you can set throughput manually, e.g. ``--throughput=150``
- __`torch_dtype`__ for BLOOM, pick `bfloat16` for Ampere (e.g. RTX 3060, A100) or newer GPUs; `float16` for other GPUs, `float32` for CPU.
- __`load_in_8_bit`__ use [LLM.8bit()](https://arxiv.org/abs/2208.07339) to fit more transformer blocks in the same memory. Remove this argument on older pre-turing GPUs or running on CPU.
- __`attn_cache_size`__ - maximum memory used for generation (and only generation). If not specified, server may run out of memory when processing too many inference queries. A good rule of thumb is 2 gb per 8 blocks. Scales proportionally.

The remaining parameters: `--host_maddrs`, `--announce_maddrs`, `--identity` and `--initial_peers` are discussed in the networking section above ("Step 1"). **When running multiple processes per server, make sure each one has a unique identity and port.**




### Step 3: Use the model

You can use test that everything works using the same interface as in README:

```python
import torch
import torch.nn.functional as F
import transformers
from src import DistributedBloomForCausalLM

initial_peers = [TODO_put_one_or_more_server_addresses_here]  # e.g. ["/ip4/127.0.0.1/tcp/more/stuff/here"]
tokenizer = transformers.BloomTokenizerFast.from_pretrained("bigscience/bloom-petals")
model = DistributedBloomForCausalLM.from_pretrained(
  "bigscience/bloom-petals", initial_peers=initial_peers, low_cpu_mem_usage=True, torch_dtype=torch.float32
)  # this model requires 14GB memory to load word embeddings (size: 14336 x 250k)


inputs = tokenizer("a cat sat", return_tensors="pt")["input_ids"]
remote_outputs = model.generate(inputs, max_length=10)
print(tokenizer.decode(remote_outputs[0]))

# "train" input embeddings by backprop through distributed transformer blocks
model.transformer.word_embeddings.weight.requires_grad = True
outputs = model.forward(input_ids=inputs)
loss = F.cross_entropy(outputs.logits.flatten(0, 1), inputs.flatten())
loss.backward()
print("Gradients (norm):", model.transformer.word_embeddings.weight.grad.norm())
```

For a more advanced usage example, please see our example on "deep" prompt-tuning here: [examples/prompt-tuning-personachat.ipynb](../blob/main/examples/prompt-tuning-personachat.ipynb).