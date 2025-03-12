from PIL import Image
import io
import msgpack
from minio import Minio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import statistics

base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, base_directory)

from utility.http import request

# Set up logging to output both to console and to a markdown (.md) file
log_file = 'execution_log_image_dataset_loader.md'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Log to file
        logging.StreamHandler(sys.stdout)  # Log to console
    ]
)

# Helper function for logging
def log_message(message):
    logging.info(message)

# Class to handle json/http requests
class HttpRequestHandler:
    def __init__(self, bucket_name, dataset_name, size):
        self.bucket_name = bucket_name
        self.dataset_name = dataset_name
        self.size = size

    def get_random_image_data(self):
        """Get a specified number of random image data (hash, UUID, file paths) using HTTP requests."""
        return request.http_get_random_image_data(self.bucket_name, self.dataset_name, self.size)
    
    def get_all_image_data(self):
        """Get all image data (hash, UUID, file paths) using HTTP requests."""
        return request.http_get_all_image_data(self.bucket_name, self.dataset_name, size=sys.maxsize)

# Class to load images from MinIO
class DatasetImageLoader:
    def __init__(self, minio_client, bucket_name):
        self.minio_client = minio_client
        self.bucket_name = bucket_name

    def get_minio_path(self, image_path):
        """Extract the MinIO path from the image path."""
        path_parts = image_path.split('/')
        minio_path = '/'.join(path_parts[1:])  # Skip the first part (bucket name)
        return minio_path

    def load_image(self, image_path):
        """Load an image from MinIO to memory."""
        minio_path = self.get_minio_path(image_path)
        try:
            response = self.minio_client.get_object(self.bucket_name, minio_path)
            image_data = response.read()
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            log_message(f"Error loading image from {minio_path}: {e}")
        return None

    def save_to_disk(self, bucket_name, dataset_name, size, save_dir=None):
        # Initialize HttpRequestHandler with the provided parameters
        request_handler = HttpRequestHandler(bucket_name, dataset_name, size)
        image_data_list = request_handler.get_random_image_data()

        item_sizes = []
        loaded_images = []
        start_time = time.time()

        # Use a default directory if none is provided
        if save_dir is None:
            save_dir = f"{bucket_name}_{dataset_name}_{size}_images"

        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        def save_image(image_data, save_dir):
            image_hash, image_uuid, image_path = image_data
            file_name = image_path.replace('/', '_')
            save_path = os.path.join(save_dir, file_name)

            loaded_image = self.load_image(image_path)

            if loaded_image:
                image_size = len(loaded_image.fp.read()) / (1024 * 1024)  # Size in MB
                item_sizes.append(image_size)
                loaded_images.append({
                        "image_hash":image_hash, 
                        "image_uuid":image_uuid, 
                        "image":loaded_image
                    })
                    # Get the image object and save to disk
                loaded_image.save(save_path)
                log_message(f"Image saved to {save_path}")
            else:
                log_message(f"Failed to load image: {image_path}")

        # Use ThreadPoolExecutor to download and save images concurrently
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_image_data = {executor.submit(save_image, image_data, save_dir): image_data for image_data in image_data_list}

            for future in as_completed(future_to_image_data):
                image_data = future_to_image_data[future]
                try:
                    future.result() 
                except Exception as e:
                    log_message(f"Error processing image {image_data}: {e}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        log_message(f"Total time save images to disk: {elapsed_time:.2f} seconds")

        # Calculate metrics
        if item_sizes:
            avg_item_size = statistics.mean(item_sizes)
            min_item_size = min(item_sizes)
            max_item_size = max(item_sizes)
            items_per_second = len(loaded_images) / elapsed_time
            m_b_per_second = sum(item_sizes) / elapsed_time
            
            log_message(f"Min image size: {min_item_size:.2f} MB")
            log_message(f"Max image size: {max_item_size:.2f} MB")
            log_message(f"Average image size: {avg_item_size:.2f} MB")
            log_message(f"Images per second: {items_per_second:.2f} items/sec")
            log_message(f"MB per second: {m_b_per_second:.2f} MB/sec")

        return elapsed_time, loaded_images

    def load_from_disk(self, local_path):
        """Load images from the local disk."""
        loaded_images = []
    
        # Check if the folder exists
        if not os.path.exists(local_path):
            log_message(f"Folder does not exist: {local_path}")
            return loaded_images

        # Loop through the files in the folder
        for file_name in os.listdir(local_path):
            file_path = os.path.join(local_path, file_name)

            # Try loading the file as an image
            try:
                image = Image.open(file_path)
                loaded_images.append({
                            "image": image
                        })
                log_message(f"Image loaded from {file_path}")
            except Exception as e:
                log_message(f"Error loading image from {file_path}: {e}")

        # Return the list of loaded images
        return loaded_images

    def load_images(self, bucket_name, dataset_name, size):
        """Load image using HttpRequestHandler."""
        request_handler = HttpRequestHandler(bucket_name, dataset_name, size)
        dataset = request_handler.get_random_image_data()

        return self.load_all_images(dataset)

    def load_all_images(self, dataset):
        """
        Load a specified number of random images using multi-threading for speed.
        Calculate the time to load data into memory and get the loading speed by dividing the number of images by time.
        """
        start_time = time.time()
        image_results = []
        image_sizes = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.load_image, image_path): (img_hash, img_uuid) for img_hash, img_uuid, image_path in dataset}

            for future in as_completed(futures):
                img_hash, img_uuid = futures[future]
                try:
                    image = future.result()
                    if image:
                        image_size = len(image.fp.read()) / (1024 * 1024)  # Size in MB
                        image_sizes.append(image_size)
                        image_results.append({
                            "image_hash": img_hash,
                            "image_uuid": img_uuid,
                            "image": image
                        })
                    else:
                        log_message(f"Failed to load image for {img_uuid} (Hash: {img_hash})")
                except Exception as e:
                    log_message(f"Error processing image {img_uuid}: {e}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        total_size = sum(image_sizes)
        load_speed = len(dataset) / elapsed_time if elapsed_time > 0 else 0

        # Calculate metrics
        if image_sizes:
            min_size = min(image_sizes)
            max_size = max(image_sizes)
            avg_size = statistics.mean(image_sizes)
            mb_per_sec = (total_size / 1024 / 1024) / elapsed_time if elapsed_time > 0 else 0
            log_message(f"Total time: {elapsed_time:.2f} seconds")
            log_message(f"Image data transfer speed: {mb_per_sec:.2f} MB/s")
            log_message(f"Image sizes: min={min_size:.2f} MB, max={max_size:.2f} MB, avg={avg_size:.2f} MB")

        return image_results, load_speed


# Class to load CLIP vectors from MinIO
class DatasetClipLoader:
    def __init__(self, minio_client, bucket_name, show_metrics=True):
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.show_metrics = show_metrics

    def get_minio_path(self, image_path):
        """Extract the MinIO path from the image path."""
        path_parts = image_path.split('/')
        minio_path = '/'.join(path_parts[1:])
        return minio_path

    def load_clip_vector(self, image_path):
        """Load a CLIP vector from MinIO to memory."""
        minio_path = self.get_minio_path(image_path)
        clip_vector_path = minio_path.replace(".jpg", "_clip_h.msgpack")
        try:
            response = self.minio_client.get_object(self.bucket_name, clip_vector_path)
            clip_data = response.read()
            clip_vector = msgpack.unpackb(clip_data)["clip-feature-vector"]
            return clip_vector
        except Exception as e:
            log_message(f"Error loading CLIP vector from {clip_vector_path}: {e}")
        return None

    def save_to_disk(self, bucket_name, dataset_name, size, save_dir=None):
        # Initialize HttpRequestHandler with the provided parameters
        request_handler = HttpRequestHandler(bucket_name, dataset_name, size)
        image_data_list = request_handler.get_random_image_data()

        item_sizes = []
        loaded_clip_vectors = []
        start_time = time.time()

        # Use a default directory if none is provided
        if save_dir is None:
            save_dir = f"{bucket_name}_{dataset_name}_{size}_clip_vectors"

        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        def save_vector(image_data, save_dir):
            image_hash, image_uuid, image_path = image_data
            file_name = image_path.replace('/', '_').replace('.jpg', '_clip_vector.msgpack')
            save_path = os.path.join(save_dir, file_name)

            loaded_vector = self.load_clip_vector(image_path)

            if loaded_vector:
                clip_size = len(msgpack.packb(loaded_vector)) / (1024 * 1024)  # Size in MB
                item_sizes.append(clip_size)
                loaded_clip_vectors.append({
                    "image_hash":image_hash, 
                    "image_uuid":image_uuid, 
                    "clip_vector":loaded_vector
                })
                with open(save_path, 'wb') as f:
                    packed_data = msgpack.packb(loaded_vector)
                    f.write(packed_data)
                log_message(f"Clip vector saved to {save_path}")
            else:
                log_message(f"Failed to load clip vector: {image_path}")

        # Use ThreadPoolExecutor to download and save clip vectors concurrently
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_image_data = {executor.submit(save_vector, image_data, save_dir): image_data for image_data in image_data_list}

            for future in as_completed(future_to_image_data):
                image_data = future_to_image_data[future]
                try:
                    future.result() 
                except Exception as e:
                    log_message(f"Error processing image {image_data}: {e}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        log_message(f"Total time save clip vectors to disk: {elapsed_time:.2f} seconds")

        # Calculate metrics
        if item_sizes and self.show_metrics:
            avg_item_size = statistics.mean(item_sizes)
            min_item_size = min(item_sizes)
            max_item_size = max(item_sizes)
            items_per_second = len(loaded_clip_vectors) / elapsed_time
            m_b_per_second = sum(item_sizes) / elapsed_time
            log_message(f"Min image size: {min_item_size:.2f} MB")
            log_message(f"Max image size: {max_item_size:.2f} MB")
            log_message(f"Average image size: {avg_item_size:.2f} MB")
            log_message(f"Clip vectors per second: {items_per_second:.2f} items/sec")
            log_message(f"MB per second: {m_b_per_second:.2f} MB/sec")

        return elapsed_time, loaded_clip_vectors

    def load_from_disk(self, local_path):
        """Load all CLIP vectors saved in .msgpack format from the specified directory into memory."""
        loaded_clip_vectors = []

        # Ensure the directory exists
        if not os.path.exists(local_path):
            log_message(f"Directory does not exist: {local_path}")
            return loaded_clip_vectors

        # Iterate through all files in the directory
        for file_name in os.listdir(local_path):
            # Only process .msgpack files
            if file_name.endswith('_clip_vector.msgpack'):
                file_path = os.path.join(local_path, file_name)

                try:
                    # Open the file and load the msgpack data
                    with open(file_path, 'rb') as f:
                        packed_data = f.read()
                        clip_vector = msgpack.unpackb(packed_data)
                        
                        loaded_clip_vectors.append({
                            "clip_vector": clip_vector
                        })

                    log_message(f"Clip vector loaded from {file_path}")
                except Exception as e:
                    log_message(f"Error loading clip vector from {file_path}: {e}")

        # Return the list of loaded clip vectors
        return loaded_clip_vectors

    def load_clip_vectors(self, bucket_name, dataset_name, size):
        """Load clip_vector using HttpRequestHandler."""
        request_handler = HttpRequestHandler(bucket_name, dataset_name, size)
        dataset = request_handler.get_random_image_data()

        return self.load_all_clip_vectors(dataset)

    def load_all_clip_vectors(self, dataset):
        """
        Load a specified number of random CLIP vectors using multi-threading for speed.
        Calculate the time to load clip vectors into memory and get the loading speed by dividing the number of clip vectors by time.
        """
        start_time = time.time()
        clip_vector_results = []
        item_sizes = []  # To track vector sizes

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.load_clip_vector, image_path): (img_hash, img_uuid) for img_hash, img_uuid, image_path in dataset}

            for future in as_completed(futures):
                img_hash, img_uuid = futures[future]
                try:
                    clip_vector = future.result()
                    if clip_vector:
                        clip_size = len(msgpack.packb(clip_vector)) / (1024 * 1024)  # Size in MB
                        item_sizes.append(clip_size)
                        clip_vector_results.append({
                            "image_hash": img_hash,
                            "image_uuid": img_uuid,
                            "clip_vector": clip_vector
                        })
                    else:
                        log_message(f"Failed to load CLIP vector for {img_uuid} (Hash: {img_hash})")
                except Exception as e:
                    log_message(f"Error processing CLIP vector {img_uuid}: {e}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        total_size = sum(item_sizes)
        load_speed = len(dataset) / elapsed_time if elapsed_time > 0 else 0

        # Calculate metrics
        if item_sizes and self.show_metrics:
            min_size = min(item_sizes)
            max_size = max(item_sizes)
            avg_size = statistics.mean(item_sizes)
            mb_per_sec = (total_size / 1024 / 1024) / elapsed_time if elapsed_time > 0 else 0
            log_message(f"Total time: {elapsed_time:.2f} seconds")
            log_message(f"Vector data transfer speed: {mb_per_sec:.2f} MB/s")
            log_message(f"Vector sizes: min={min_size:.2f} MB, max={max_size:.2f} MB, avg={avg_size:.2f} MB")
            
        return clip_vector_results, load_speed

# Class to load vae latents from MinIO
class DatasetVaeLoader:
    def __init__(self, minio_client, bucket_name, show_metrics= True):
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.show_metrics = show_metrics

    def get_minio_path(self, image_path):
        """Extract the MinIO path from the image path."""
        path_parts = image_path.split('/')
        minio_path = '/'.join(path_parts[1:])
        return minio_path

    def load_vae_latent(self, image_path, resolution=512):
        """Load a vae latent from MinIO to memory."""
        minio_path = self.get_minio_path(image_path)
        if resolution == 512:
            vae_latent_path = minio_path.replace(".jpg", "_vae_latent.msgpack")
        else:
            vae_latent_path = minio_path.replace(".jpg", f"_vae_latent_{resolution}.msgpack")

        try:
            response = self.minio_client.get_object(self.bucket_name, vae_latent_path)
            vae_data = response.read()
            vae_latent = msgpack.unpackb(vae_data)["latent_vector"]
            return vae_latent
        except Exception as e:
            log_message(f"Error loading CLIP vector from {vae_latent_path}: {e}")
        return None

    def save_to_disk(self, bucket_name, dataset_name, size, save_dir=None):
        # Initialize HttpRequestHandler with the provided parameters
        request_handler = HttpRequestHandler(bucket_name, dataset_name, size)
        image_data_list = request_handler.get_random_image_data()

        item_sizes = []
        loaded_vae_latents = []
        start_time = time.time()

        # Use a default directory if none is provided
        if save_dir is None:
            save_dir = f"{bucket_name}_{dataset_name}_{size}_vae_latents"

        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        def save_vae_latent(image_data, save_dir):
            image_hash, image_uuid, image_path = image_data
            file_name = image_path.replace('/', '_').replace('.jpg', '_vae_latent.msgpack')
            save_path = os.path.join(save_dir, file_name)

            loaded_vae_latent = self.load_vae_latent(image_path)

            if loaded_vae_latent:
                clip_size = len(msgpack.packb(loaded_vae_latent)) / (1024 * 1024)  # Size in MB
                item_sizes.append(clip_size)
                loaded_vae_latents.append({
                    "image_hash":image_hash, 
                    "image_uuid":image_uuid, 
                    "vae_latent":loaded_vae_latent
                })
                with open(save_path, 'wb') as f:
                    packed_data = msgpack.packb(loaded_vae_latent)
                    f.write(packed_data)
                log_message(f"Vae latent saved to {save_path}")
            else:
                log_message(f"Failed to load vae latent: {image_path}")

        # Use ThreadPoolExecutor to download and save vae latents concurrently
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_vae_data = {executor.submit(save_vae_latent, image_data, save_dir): image_data for image_data in image_data_list}

            for future in as_completed(future_to_vae_data):
                image_data = future_to_vae_data[future]
                try:
                    future.result() 
                except Exception as e:
                    log_message(f"Error processing image {image_data}: {e}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        log_message(f"Total time save vae latents to disk: {elapsed_time:.2f} seconds")

        # Calculate metrics
        if item_sizes and self.show_metrics:
            avg_item_size = statistics.mean(item_sizes)
            min_item_size = min(item_sizes)
            max_item_size = max(item_sizes)
            items_per_second = len(loaded_vae_latents) / elapsed_time
            m_b_per_second = sum(item_sizes) / elapsed_time
            
            log_message(f"Min image size: {min_item_size:.2f} MB")
            log_message(f"Max image size: {max_item_size:.2f} MB")
            log_message(f"Average image size: {avg_item_size:.2f} MB")
            log_message(f"Vae latents per second: {items_per_second:.2f} items/sec")
            log_message(f"MB per second: {m_b_per_second:.2f} MB/sec")

        return elapsed_time, loaded_vae_latents

    def load_from_disk(self, local_path):
        """Load all Vae latents saved in .msgpack format from the specified directory into memory."""
        loaded_vae_latents = []

        # Ensure the directory exists
        if not os.path.exists(local_path):
            log_message(f"Directory does not exist: {local_path}")
            return loaded_vae_latents

        # Iterate through all files in the directory
        for file_name in os.listdir(local_path):
            # Only process .msgpack files
            if file_name.endswith('_vae_latent.msgpack'):
                file_path = os.path.join(local_path, file_name)

                try:
                    # Open the file and load the msgpack data
                    with open(file_path, 'rb') as f:
                        packed_data = f.read()
                        vae_latent = msgpack.unpackb(packed_data)
                        
                        loaded_vae_latents.append({
                            "vae_latent": vae_latent
                        })

                    log_message(f"Vae latent loaded from {file_path}")
                except Exception as e:
                    log_message(f"Error loading vae latent from {file_path}: {e}")

        # Return the list of loaded vae latents
        return loaded_vae_latents

    def load_vae_latents(self, bucket_name, dataset_name, size):
        """Load vae_latent using HttpRequestHandler."""
        request_handler = HttpRequestHandler(bucket_name, dataset_name, size)
        dataset = request_handler.get_random_image_data()

        return self.load_all_vae_latents(dataset)

    def load_all_vae_latents(self, dataset, resolution=512):
        """
        Load a specified number of random vae latents using multi-threading for speed.
        Calculate the time to load vae latents into memory and get the loading speed by dividing the number of clip vectors by time.
        """
        start_time = time.time()
        vae_latent_results = []
        item_sizes = []  # To track vector sizes

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.load_vae_latent, image_path, resolution): (img_hash, img_uuid) for img_hash, img_uuid, image_path in dataset}

            for future in as_completed(futures):
                img_hash, img_uuid = futures[future]
                try:
                    vae_latent = future.result()
                    if vae_latent:
                        clip_size = len(msgpack.packb(vae_latent)) / (1024 * 1024)  # Size in MB
                        item_sizes.append(clip_size)
                        vae_latent_results.append({
                            "image_hash": img_hash,
                            "image_uuid": img_uuid,
                            "vae_latent": vae_latent
                        })
                    else:
                        log_message(f"Failed to load vae latent for {img_uuid} (Hash: {img_hash})")
                except Exception as e:
                    log_message(f"Error processing vae latent {img_uuid}: {e}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        total_size = sum(item_sizes)
        load_speed = len(dataset) / elapsed_time if elapsed_time > 0 else 0

        # Calculate metrics
        if item_sizes and self.show_metrics:
            min_size = min(item_sizes)
            max_size = max(item_sizes)
            avg_size = statistics.mean(item_sizes)
            mb_per_sec = (total_size / 1024 / 1024) / elapsed_time if elapsed_time > 0 else 0
            log_message(f"Total time: {elapsed_time:.2f} seconds")
            log_message(f"Vae vector data transfer speed: {mb_per_sec:.2f} MB/s")
            log_message(f"Vae vector sizes: min={min_size:.2f} MB, max={max_size:.2f} MB, avg={avg_size:.2f} MB")
            
        return vae_latent_results, load_speed

# Class to store images, hashes, UUIDs, file paths, and clip vectors
class Receiver:
    def __init__(self):
        self.data = []

    def store_data(self, img_hash, img_uuid, image_path, image, clip_vector, vae_latent):
        """Store image data and clip vector data."""
        self.data.append({
            'hash': img_hash,
            'uuid': img_uuid,
            'image_path': image_path,
            'image': image,
            'clip_vector': clip_vector,
            'vae_latent': vae_latent
        })
    
    def store_combined_data(self, dataset, images, clip_vectors, vae_latents):
        """Store combined image and clip vector data."""
        # Create a mapping from (hash, uuid) to clip vector
        clip_vector_dict = {
            (clip['image_hash'], clip['image_uuid']): clip['clip_vector'] for clip in clip_vectors
        }

        vae_latent_dict = {
            (vae['image_hash'], vae['image_uuid']): vae['vae_latent'] for vae in vae_latents
        }

        # Store combined data
        for image_data in images:
            img_hash = image_data['image_hash']
            img_uuid = image_data['image_uuid']
            image = image_data['image']
            image_path = next((path for h, u, path in dataset if h == img_hash and u == img_uuid), None)
            clip_vector = clip_vector_dict.get((img_hash, img_uuid))
            vae_latent = vae_latent_dict.get((img_hash, img_uuid))
            self.store_data(img_hash, img_uuid, image_path, image, clip_vector, vae_latent)

    def get_data(self):
        """Return the stored data."""
        return self.data


# Example usage
if __name__ == "__main__":
    # Initialize MinIO client
    minio_client = Minio(
        "103.20.60.90:9000",
        # access_key="access_key",
        # secret_key="secret_key",

        secure=False
    )

    # First example usage
    # Initialize ImageLoaderByTag and VectorLoaderByTag
    image_loader = DatasetImageLoader(minio_client, bucket_name="extracts")
    vector_loader = DatasetClipLoader(minio_client, bucket_name="extracts")
    vae_loader = DatasetVaeLoader(minio_client, bucket_name="extracts")

    loaded_images, image_loading_time = image_loader.load_images(bucket_name="extracts", dataset_name="axiom-verge", size=1)
    print(f"Successfully loaded {len(loaded_images)} images in {image_loading_time:.2f} seconds.")
    for img_info in loaded_images:
        print(f"Image UUID: {img_info['image_uuid']}, Image Hash: {img_info['image_hash']}")
        img_info['image'].show()  

    # loaded_vectors, vector_loading_time = vector_loader.load_clip_vectors(bucket_name="extracts", dataset_name="axiom-verge", size=1)
    # print(f"Successfully loaded {len(loaded_vectors)} CLIP vectors in {vector_loading_time:.2f} seconds.")
    # for vec_info in loaded_vectors:
    #     print(f"Image UUID: {vec_info['image_uuid']}, Image Hash: {vec_info['image_hash']}")

    # loaded_vae_latents, vae_latents_loading_time = vae_loader.load_vae_latents(bucket_name="extracts", dataset_name="axiom-verge", size=1)
    # print(f"Successfully loaded {len(loaded_vae_latents)} vae latents in {vae_latents_loading_time:.2f} seconds.")
    # for vae_info in loaded_vae_latents:
    #     print(f"Loaded vae latent for Image UUID: {vae_info['image_uuid']}, Image Hash: {vae_info['image_hash']}")

    # # Second example usage
    # image_loader.save_to_disk(bucket_name="extracts", dataset_name="axiom-verge", size=2, save_dir="E:\\Skycoin\\test_download_image_from_minio_output\\extracts_axiom-verge_images")
    # image_loader.save_to_disk(bucket_name="extracts", dataset_name="axiom-verge", size=1, save_dir=None)
    # images = image_loader.load_from_disk(local_path='E:\\Skycoin\\test_download_image_from_minio_output\\extracts_axiom-verge_images')
    # print(f"Total images loaded: {len(images)}")
    # for image_data in images:
    #     image = image_data['image']
    #     image.show()  

    # vector_loader.save_to_disk(bucket_name="extracts", dataset_name="axiom-verge", size=2, save_dir="E:\\Skycoin\\test_download_image_from_minio_output\\extracts_axiom-verge_clip_vectors")
    # vector_loader.save_to_disk(bucket_name="extracts", dataset_name="axiom-verge", size=2, save_dir=None)
    # clip_vectors = vector_loader.load_from_disk(local_path='E:\\Skycoin\\test_download_image_from_minio_output\\extracts_axiom-verge_clip_vectors')
    # print(f"Total CLIP vectors loaded: {len(clip_vectors)}")
    # for vector_data in clip_vectors:
    #     clip_vector = vector_data['clip_vector']
    #     print(f"Loaded CLIP vector: {clip_vector}")

    # vae_loader.save_to_disk(bucket_name="extracts", dataset_name="axiom-verge", size=2, save_dir="E:\\Skycoin\\test_download_image_from_minio_output\\extracts_axiom-verge_vae_latents")
    # vae_loader.save_to_disk(bucket_name="extracts", dataset_name="axiom-verge", size=2, save_dir=None)
    # vae_latents = vae_loader.load_from_disk(local_path='E:\\Skycoin\\test_download_image_from_minio_output\\extracts_axiom-verge_vae_latents')
    # print(f"Total vae latents loaded: {len(vae_latents)}")
    # for vae_data in vae_latents:
    #     vae_latent = vae_data['vae_latent']
    #     print(f"Loaded CLIP vector: {vae_latent}")

    # # Third example usage
    # # 1. Get the dataset using HttpRequestHandler
    # http_handler = HttpRequestHandler(bucket_name="extracts", dataset_name="axiom-verge", size=1)
    # dataset = http_handler.get_random_image_data()

    # # 2. Create a Receiver instance
    # receiver = Receiver()

    # # 3. Load a specified number of random images and related data from MinIO
    # image_loader = DatasetImageLoader(minio_client, bucket_name="extracts")
    # images, image_load_speed = image_loader.load_all_images(dataset)
    # for image_data in images:
    #     print(f"Loaded image {image_data['image_uuid']} (Hash: {image_data['image_hash']})")
    # print(f"Loaded {len(images)} images at {image_load_speed:.2f} images/second.")

    # # 4. Load a specified number of random images of the clip vector and related data from MinIO
    # vector_loader = DatasetClipLoader(minio_client, bucket_name="extracts")
    # clip_vectors, clip_vector_load_speed = vector_loader.load_all_clip_vectors(dataset)
    # for clip_vector_data  in clip_vectors:
    #     print(f"Loaded CLIP vector {clip_vector_data['clip_vector']} for {clip_vector_data['image_uuid']} (Hash: {clip_vector_data['image_hash']})")
    # print(f"Loaded {len(clip_vectors)} CLIP vectors at {clip_vector_load_speed:.2f} vectors/second.")

    # # 5. Load a specified number of random images of the vae latent and related data from MinIO
    # vae_loader = DatasetVaeLoader(minio_client, bucket_name="extracts")
    # vae_latents, vae_latent_load_speed = vae_loader.load_all_vae_latents(dataset)
    # for vae_latent_data  in vae_latents:
    #     print(f"Loaded vae latent {vae_latent_data['vae_latent']} for {vae_latent_data['image_uuid']} (Hash: {vae_latent_data['image_hash']})")
    # print(f"Loaded {len(vae_latents)} vae latent at {vae_latent_load_speed:.2f} vectors/second.")

    # # 6. Store combined images and clip vectors and vae latent in Receiver
    # receiver.store_combined_data(dataset, images, clip_vectors, vae_latents)

    # # Now receiver has all data stored
    # data = receiver.get_data()
    # for item in data:
    #     clip_vector_str = str(item['clip_vector'])[:10]
    #     vae_latent_str = str(item['vae_latent'])[:10]
    #     print(f"Stored data for image {item['uuid']} (Hash: {item['hash']}), clip vector: {clip_vector_str}, vae latent: {vae_latent_str}.")  

