from huggingface_hub import snapshot_download

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

snapshot_download(
    repo_id=model_name,
    local_dir="./models/Qwen2.5-0.5B-Instruct"
)

print("Model downloaded successfully.")