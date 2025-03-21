import io
import re
import pandas as pd

from utility.minio import minio_manager
from utility.path import separate_bucket_and_file_path


def save_loss_per_image(minio_client, output_directory, loss_per_image, num_checkpoint):
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(list(loss_per_image.items()), columns=['Image Hash', 'Loss'])

    # Convert DataFrame to CSV
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)

    # Reset buffer position to the beginning after writing
    csv_buffer.seek(0)

    # Upload the CSV file to MinIO
    bucket, output_directory= separate_bucket_and_file_path(output_directory)
    minio_path= output_directory + f"/reports/processed_images/losses_per_image_checkpoint_{num_checkpoint}.csv"
    minio_manager.upload_data(minio_client, bucket, minio_path, csv_buffer)

def get_all_image_hashes(minio_client, output_directory):
    image_hashes = set()

    # Get folder path where previously processed images are stored
    bucket, output_directory = separate_bucket_and_file_path(output_directory)
    folder_path = output_directory + f"/reports/processed_images/"

    # List all objects in the specified folder
    objects = list(minio_client.list_objects(bucket, prefix=folder_path, recursive=True))
    
    # Filter and sort CSV files based on checkpoint numbers
    csv_files = [
        obj.object_name for obj in objects if obj.object_name.endswith('.csv')
    ]
    csv_files_sorted = sorted(csv_files, key=lambda x: int(re.search(r'checkpoint_(\d+).csv', x).group(1)))

    for file_path in csv_files_sorted:
        # Fetch CSV file content
        response = minio_client.get_object(bucket, file_path)
        csv_data = response.read()
        
        # Read CSV into a DataFrame
        df = pd.read_csv(io.BytesIO(csv_data))
        
        # Check for 'Image Hash' column
        if 'Image Hash' in df.columns:
            for hash_value in df['Image Hash'].unique():
                if hash_value in image_hashes:
                    # If duplicate found, reset set and continue with the next files
                    image_hashes.clear()
                    print(f"Duplicate hash '{hash_value}' found in checkpoint {file_path}. Resetting the set and continuing...")
                image_hashes.add(hash_value)

    return image_hashes