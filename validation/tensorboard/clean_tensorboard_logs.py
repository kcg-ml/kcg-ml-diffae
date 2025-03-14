import argparse
import os, sys
import pandas as pd
import tensorflow as tf
from minio import Minio

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from utility.minio import minio_manager
from utility.model_cards.model_card import ModelCard

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect tensorboard event file content for specific models")

    # minio keys
    parser.add_argument('--minio-access-key', type=str, required=True, help='Access key for model MinIO storage.')
    parser.add_argument('--minio-secret-key', type=str, required=True, help='Secret key for model MinIO storage.')

    # Model id
    parser.add_argument('--model-id', required=True, type=int, help='Id of the model', default=None)
    parser.add_argument('--num-checkpoint', required=True, type=int, help='Id of the model', default=None)
    parser.add_argument('--log-dir', type=str, help='directory where tensorboard logs are stored', default="./output")

    return parser.parse_args()

def prune_event_file(input_file, output_file, cutoff_step, cutoff_k_images):

    with tf.io.TFRecordWriter(output_file) as writer:
        for event in tf.compat.v1.train.summary_iterator(input_file):
            skip_event = False
            if event.HasField("summary"):
                for value in event.summary.value:
                    if value.tag == "Loss/Step" and event.step > cutoff_step:
                        skip_event = True
                        break
                    if value.tag == "Loss/k_images" and event.step > cutoff_k_images:
                        skip_event = True
                        break
            if not skip_event:
                writer.write(event.SerializeToString())

def prune_event_files(model_id, log_dir, cutoff_step, cutoff_k_images):
    """
    Searches for all TensorBoard event files under the specified log_dir and model_id,
    prunes events exceeding the cutoff for both 'Loss/Step' and 'Loss/k_images',
    and overwrites the original event files with the pruned files.
    """
    pattern = f"{log_dir}/tensorboard_logs/{model_id}/events.out.tfevents.*"
    event_files = tf.io.gfile.glob(pattern)
    for event_file in event_files:
        temp_file = event_file + ".tmp"
        print(f"Pruning file: {event_file}")
        prune_event_file(event_file, temp_file, cutoff_step, cutoff_k_images)
        # Overwrite the original file with the pruned file.
        tf.io.gfile.rename(temp_file, event_file, overwrite=True)
        print(f"Overwritten {event_file} with pruned events.")

def check_tensorboard_logs(model_id, log_dir):
    """
    Reads a TensorBoard event file and prints all recorded steps.
    """
    tensorboard_log_dir = f"{log_dir}/tensorboard_logs/{model_id}"
    event_file = sorted(os.listdir(tensorboard_log_dir))[-1]  # Get latest event file
    event_filepath = os.path.join(tensorboard_log_dir, event_file)
    print(f"Checking TensorBoard file: {event_filepath}")

    try:
        steps = []
        for record in tf.data.TFRecordDataset(event_filepath):
            event = tf.compat.v1.Event()
            event.ParseFromString(record.numpy())
            if event.HasField("summary"):
                steps.append(event.step)

        print(f"Total Records Found: {len(steps)}")

    except Exception as e:
        print(f"Error reading TensorBoard file: {e}")

def get_last_checkpoint_step(minio_client: Minio, model_id: int, checkpoint: int):
    print("loading checkpoint steps")
    # load checkpoint info
    model_card= ModelCard(minio_client)
    model_info= model_card.load_checkpoint_model_card("diffae", model_id, checkpoint)
    # get batch size and checkpointing steps
    current_step= model_info["step"]
    current_k_images = model_info["k_images"]
    
    return current_step, current_k_images

def main():
    args = parse_args()
    
    # get minio client
    minio_client = minio_manager.get_minio_client(minio_ip_addr= "192.168.3.6:9000",
                                        minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)
    
    # get checkpoint step and k_images
    step, k_images = get_last_checkpoint_step(minio_client, args.model_id, args.num_checkpoint)
    # clean event files
    prune_event_files(args.model_id, args.log_dir, step, k_images)

if __name__== "__main__":
    main()