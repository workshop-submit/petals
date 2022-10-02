To run Petals servers with your own model, you need to convert the model weights into a Petals-compatible format.
This conversion splits each individual block into a separate branch. This allows each peer to download only the
layers they need, instead of the entire 350GB model.

For BLOOM models, you can convert them using the following script:
```bash
# convert model from HF hub to a distributed format (can take hours depending on your connection!)
MY_WRITE_TOKEN=TODO_WRITE_TOKEN_FROM_https://huggingface.co/settings/token
python -m cli.convert_model --model bigscience/bloom-6b3  \
  --output_path ./converted_model --output_repo bigscience/test-bloomd-6b3 \
  --use_auth_token $MY_WRITE_TOKEN  # ^-- todo replace output repo with something you have access to
```

If you want to run a non-BLOOM model (e.g. [OPT](https://arxiv.org/abs/2205.01068) or [YALM](https://github.com/yandex/YaLM-100B)),
you will need to edit the code a bit.
Currently, Petals uses a vanilla implementation of BLOOM in `src/bloom`, so it is possible to replace it with other models from Hugging Face transformers. 

Assuming your model is already is compatible with Hugging Face, you will need 3 extra steps:

1. Edit `cli/convert_model.py` to partition your model checkpoint into individual blocks and non-transformer layers.
   Once you are done, run this script to convert your model and upload it to Hugging Face. If your model is private,
   you can use your internal storage instead (see next step).
2. In `src/bloom/from_pretrained.py`, edit `load_pretrained_block` to load a single block of your custom model.
  Your block should be able to run `.forward(hidden_states=..., use_cache=true_or_false, layer_past=optional_tensors)`.
  After this step, you should be able to launch a server with the new model name.
3. Open `src/client/remote_model.py` and change `DistributedBloomModel` to load the model of your choice.
  Create non-transformer layers (e.g. embeddings and logits) as usual. Instead of loading transformer blocks,
  create a RemoteSequential instance. 

Once you are done, run `tests/test_full_model.py` to verify that your conversion went correctly.
In future, we hope to streamline this process, making it possible to serve any language model available on Hugging Face.
If you with this future to come sooner and willing to work on a pull-request, please contact us via issues.

