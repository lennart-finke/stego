#!/bin/bash
set -euo pipefail

# Create timestamp for log directory
timestamp=$(date '+%Y%m%d_%H%M%S')

# Build arrays of full model IDs for encoders and decoders
encoder_models=(openai/o3-mini)
decoder_models=(openai/o3-mini)
# Example additional models that could be added:
# "openai/o3-mini", "anthropic/claude-3-7-sonnet-latest", "openai/gpt-4o", "anthropic/claude-3-opus-latest"

# Create the base log directory with timestamp
base_log_dir="logs/${timestamp}"
mkdir -p "$base_log_dir"

# Function to sanitize model name for use in file paths
sanitize_model_name() {
    local model=$1
    # Replace colons and other problematic characters with underscores
    echo "$model" | sed 's/[:]/_/g'
}

# Function to create config file and return the tmux command
create_tmux_command() {
  local test_model=$1
  
  # Compute a short name for test_model (everything after the last slash).
  test_short="${test_model##*/}"
  # Sanitize the short name for use in file paths
  test_short=$(sanitize_model_name "$test_short")
  
  # Build decoder models list
  decoder_models_list=""
  for model in "${decoder_models[@]}"; do
    if [ -z "$decoder_models_list" ]; then
      decoder_models_list=\"$model\"
    else
      decoder_models_list="$decoder_models_list, \"$model\""
    fi
  done
  
  # Build a log directory name from the test model short name
  log_dir="${base_log_dir}/${test_short}_all_decoders"
  mkdir -p "$log_dir"
  
  # Create config file
  config_file="${log_dir}/task_config.json"
  cat > "$config_file" << EOF
{
  "decoder_models": [${decoder_models_list}],
  "encoder_model": "${test_model}"
}
EOF
  
  # Create the command string
  echo "inspect eval steganography.py --model $test_model --limit 10 --task-config \"$config_file\" --max-connections 20 --log-format json --log-dir \"$log_dir\" --temperature 0.1 --reasoning-tokens 2024 --max-tokens 8192"
}

# Create a new tmux session
tmux_session="parallel_eval_${timestamp}"
tmux new-session -d -s "$tmux_session"

# Calculate window layout based on number of encoder models
num_models=${#encoder_models[@]}

# Start with first model in the initial pane
first_model=${encoder_models[0]}
first_command=$(create_tmux_command "$first_model")
tmux send-keys -t "$tmux_session" "echo \"Running model: $first_model\"" C-m
tmux send-keys -t "$tmux_session" "$first_command" C-m

# Add remaining models in separate panes
for ((i=1; i<num_models; i++)); do
  model=${encoder_models[$i]}
  command=$(create_tmux_command "$model")
  
  # Split window and send command
  tmux split-window -t "$tmux_session"
  tmux select-layout -t "$tmux_session" tiled
  
  tmux send-keys -t "$tmux_session:0.$i" "echo \"Running model: $model\"" C-m
  tmux send-keys -t "$tmux_session:0.$i" "$command" C-m
done

# Set even layout
tmux select-layout -t "$tmux_session" tiled

# Print instructions for the user
echo "Started parallel execution in tmux session: $tmux_session"
echo "To attach to the session: tmux attach-session -t $tmux_session"
echo "To detach from session: press Ctrl+B then D"
echo "To kill the session when done: tmux kill-session -t $tmux_session"