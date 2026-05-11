# SWIFT

**SWIFT: Prompt-Adaptive Memory for Efficient Interactive Long Video Generation**

SWIFT is an efficient interactive long-video generation framework for multi-prompt long video synthesis. It is designed for streaming / autoregressive text-to-video generation where prompts change over time, aiming to preserve temporal coherence while adapting quickly to new semantic instructions.

## Highlights

- Interactive multi-prompt video generation
- Prompt-adaptive memory mechanism for smooth semantic transitions
- Semantic Injection Cache for lightweight prompt switching
- Adaptive Dynamic Window for efficient long-horizon inference
- Supports configurable generation lengths through YAML config files
- Supports custom prompt datasets in JSONL format

## Repository Structure

```text
SWIFT/
├── configs/                         # Configuration files
│   ├── 30s_config.yaml              # Example config for 30-second generation
│   └── longlive_interactive_inference.yaml
├── prompt/
│   └── interactive_benchmark.jsonl  # Example interactive prompt dataset
├── videos/                          # Output directory for generated videos
├── interactive_inference.py         # Interactive inference entry point
├── interactivate_inference.sh       # Shell script for interactive inference
├── inference.py
├── requirements.txt
└── README.md
```

## Installation

Clone this repository:

```bash
git clone https://github.com/ShanwenTan/SWIFT.git
cd SWIFT
```

Create and activate a Python environment:

```bash
conda create -n swift python=3.10 -y
conda activate swift
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Checkpoints

Before running inference, please make sure the required model checkpoints are placed at the paths specified in the YAML config file.

For example, the config file may contain checkpoint paths such as:

```yaml
generator_ckpt: models/longlive_models/models/longlive_base.pt
lora_ckpt: models/longlive_models/models/lora.pt
```

If your checkpoints are stored in different directories, update the corresponding fields in the config file before running inference.

## Quick Start

Run interactive inference with:

```bash
bash interactive_inference.sh
```

The shell script runs interactive video generation using the configuration file specified by `--config_path`.

A typical command looks like:

```bash
torchrun \
  --nproc_per_node=1 \
  --master_port=29500 \
  interactive_inference.py \
  --config_path configs/longlive_interactive_inference.yaml
```

Generated videos will be saved to the output directory specified by `output_folder` in the config file.

## Run Videos with Different Durations

If you want to run videos with different durations, replace the configuration file path under `configs/`.

For example, to run the 30-second configuration, change the config path to:

```bash
torchrun \
  --nproc_per_node=1 \
  --master_port=29500 \
  interactive_inference.py \
  --config_path configs/30s_config.yaml
```

You can also modify `interactivate_inference.sh` directly:

```bash
--config_path configs/30s_config.yaml
```

Similarly, you can create your own config files, such as:

```text
configs/45s_config.yaml
configs/60s_config.yaml
configs/90s_config.yaml
```

Then run inference by replacing the config path:

```bash
torchrun \
  --nproc_per_node=1 \
  --master_port=29500 \
  interactive_inference.py \
  --config_path configs/90s_config.yaml
```

## Custom Prompts

If you want to run SWIFT with your own prompts, construct your own JSONL dataset following the same format as:

```text
prompt/interactive_benchmark.jsonl
```

Each line in the JSONL file should be one independent video sample. The sample should contain a `prompts` field, where `prompts` is a list of text prompts describing the semantic stages of the video.

Example:

```json
{"prompts": ["A young wizard is practicing magic in an ancient stone chamber.", "A small green dragon flies into the chamber and lands on the wizard's shoulder.", "The wizard and the dragon practice magic together.", "The dragon sneezes and sends harmless sparks into the air.", "The wizard and dragon create a glowing ball of light.", "The glowing ball bursts into magical stars."]}
```

After creating your own prompt file, update the `data_path` field in the config file:

```yaml
data_path: prompt/my_prompts.jsonl
```

For example:

```yaml
data_path: prompt/interactive_benchmark.jsonl
output_folder: videos/my_interactive_results
num_output_frames: 240
switch_frame_indices: 40, 80, 120, 160, 200
```

Important notes:

- Each line in the JSONL file corresponds to one generated video.
- The number of `switch_frame_indices` should be equal to the number of prompt segments minus one.
- For six prompts, use five switch indices.
- Make sure the prompt schedule matches the target video length.

## Example Config Fields

Common fields in the YAML config include:

```yaml
data_path: prompt/interactive_benchmark.jsonl
output_folder: videos/interactive_60s
num_output_frames: 240
switch_frame_indices: 40, 80, 120, 160, 200
seed: 1
num_samples: 1
```

For a 30-second setting, the config may use:

```yaml
num_output_frames: 120
switch_frame_indices: 20, 40, 60, 80, 100
```

## Output

After inference, generated videos will be saved under the folder specified by:

```yaml
output_folder: videos/interactive_60s
```

The output video filenames are automatically generated during inference.


## License

This project is released under the Apache-2.0 License. See [LICENSE](LICENSE) for details.


