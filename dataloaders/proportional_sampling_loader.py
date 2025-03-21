import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import os
import random
import sys
from minio import Minio
import torch
from tqdm import tqdm
import msgpack
from concurrent.futures import ThreadPoolExecutor, as_completed

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())
from utility.http import request
from utility.minio import minio_manager
from utility.path import separate_bucket_and_file_path

EXTRACTS_BUCKET= "extracts"


class ProportionalSamplingLoader:
    def __init__(self,
                 minio_client: Minio,
                 model_type: str = "elm",
                 image_resolution: int = 512):
        
        self.minio_client= minio_client
        self.model_type= model_type
        self.image_resolution = image_resolution

        self.tag_scores = {}
        self.used_image_hashes= set()
        self.unique_images= []

    def load_classifiers(self):
        print("Loading classifier data")
        tag_list= request.http_get_tag_list()
        classifier_list = request.http_get_classifier_model_list()
        tag_id_dict={}
        classifiers=[]

        for tag in tag_list:
            tag_id_dict[tag['tag_id']] = tag['tag_string']

        exclude_keywords = ["all_resolutions", "defect", "irrelevant", "test"]

        for classifier in tqdm(classifier_list):
            classifier_name = classifier["classifier_name"]

            if self.model_type in classifier_name and all(keyword not in classifier_name for keyword in exclude_keywords):
                classifiers.append(classifier)

        return classifiers, tag_id_dict

    def create_sampling_datasets(self, batch_size=300, classifier_threshold=0.6):
        # Load list of classifiers and tags
        classifiers, tag_id_dict = self.load_classifiers()

        # Initialize image details cache
        print("Loading scores for each tag")
        image_details_cache = {}  # Cache for storing image details

        for classifier in tqdm(classifiers):
            print(f"{len(image_details_cache)} image file paths are currently cached \n")

            tag_id = classifier["tag_id"]
            tag_name = tag_id_dict[tag_id]

            # Proceed to load classifier scores and process them
            print(f"Loading classifier scores for tag: {tag_name} \n")
            scores = request.http_get_images_by_tag_by_batches(tag_name, self.model_type, classifier_threshold, None, "extract_image")

            print(f"We have {len(scores)} scores \n")
            tag_scores_data = {}
            image_hash_batch = []

            for score in scores:
                if score["score"] < classifier_threshold:
                    continue

                image_hash = score["image_hash"]
                # Populate score data
                score_data = {
                    "image_hash": image_hash,
                    "score": round(score["score"], 2)
                }

                tag_scores_data[image_hash] = score_data
                
                file_path= score.get("file_path", None)
                if file_path:
                    # Directly retrieve from api response and store in cache
                    tag_scores_data[image_hash]['file_path'] = file_path
                    image_details_cache[image_hash] = file_path

                if image_hash in image_details_cache:
                    # Directly retrieve from cache
                    tag_scores_data[image_hash]['file_path'] = image_details_cache[image_hash]
                else:
                    # Collect image hashes in the batch if they are not cached yet
                    image_hash_batch.append(image_hash)

                if len(image_hash_batch) >= batch_size:
                    self.fetch_and_update_image_details(image_hash_batch, tag_scores_data, image_details_cache)
                    image_hash_batch = []

            if len(image_hash_batch) > 0:
                self.fetch_and_update_image_details(image_hash_batch, tag_scores_data, image_details_cache)
                image_hash_batch = []

            # After processing each tag, create and upload the CSV file for that tag
            self.create_and_upload_csv_file_for_tag(tag_name, tag_scores_data)

        print("Finished loading and uploading all tag scores.")

    def fetch_and_update_image_details(self, image_hash_batch, tag_scores_data, image_details_cache):
        image_data = request.http_get_image_details_by_hashes(image_hash_batch, ["image_hash", "file_path"])

        # Store image details in the cache for future use
        for image in image_data:
            image_hash = image["image_hash"]
            file_path = image.get('file_path')
            if file_path:
                image_details_cache[image_hash] = file_path  # Cache the file path for this image hash
                tag_scores_data[image_hash]['file_path'] = file_path

    def create_and_upload_csv_file_for_tag(self, tag_name, tag_scores_data):
        print(f"Uploading CSV file for tag: {tag_name}")

         # Sort scores by the 'score' field in descending order
        sorted_scores = sorted(tag_scores_data.values(), key=lambda x: x['score'], reverse=True)

        # Create a StringIO object to write CSV data
        csv_buffer = io.StringIO()
        fieldnames = sorted_scores[0].keys() if sorted_scores else []
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_scores)

        # Convert StringIO to bytes
        csv_data = csv_buffer.getvalue().encode('utf-8')

        # Create a BytesIO object from the CSV data
        csv_bytes_io = io.BytesIO(csv_data)
        csv_bytes_io.seek(0)  # Move to the beginning of the BytesIO object

        # Define the object name in Minio
        object_name = f"kandinsky/proportional_sampling/{tag_name}.csv"

        # Upload the file
        minio_manager.upload_data(self.minio_client, "models", object_name, csv_bytes_io)

    # Function to load only necessary columns from a CSV file
    def load_tag_score_dict(self, csv_file, tag_name):
        # Define a list to store filtered data
        scores = []

        # Stream CSV from Minio and filter as it reads each row
        response = self.minio_client.get_object("models", csv_file)
        with io.TextIOWrapper(io.BytesIO(response.read()), encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                scores.append({
                    "image_hash": row["image_hash"],
                    "file_path": row["file_path"],
                })
        
        # Store filtered scores in tag_scores dictionary by tag_name
        self.tag_scores[tag_name] = scores

        return scores

    def load_tag_scores(self, prefixes=None):
        print("Loading filtered tag scores...")
        objects = self.minio_client.list_objects("models", prefix="kandinsky/proportional_sampling/")
        csv_files = [obj.object_name for obj in objects if obj.object_name.endswith('.csv')]

        unique_hashes = set()  # Set to store unique image hashes
        num_tags = 0

        # Load each CSV file with filtering
        with tqdm(total=len(csv_files)) as pbar:
            for csv_file in csv_files:
                tag_name = csv_file.replace('.csv', '').split("/")[-1]
                if prefixes is None or any(tag_name.startswith(prefix) for prefix in prefixes):
                    try:
                        # Load and filter tag scores
                        filtered_scores = self.load_tag_score_dict(csv_file, tag_name)
                        num_tags += 1
                        
                        # Add image hashes to unique_hashes set
                        for score in filtered_scores:
                            image_hash= score["image_hash"]
                            if image_hash not in unique_hashes:
                                # add new hash to unique image hash list
                                unique_hashes.add(image_hash)
                                # add the image data
                                self.unique_images.append({
                                    "image_hash": score["image_hash"],
                                    "file_path": score["file_path"],
                                    "tag": tag_name,
                                })
                    except Exception as e:
                        print(f"Failed to load tag scores for '{tag_name}': {e}")
                pbar.update(1)
        
        num_images= len(unique_hashes)
        del unique_hashes

        # Return the number of tags and the count of unique image hashes
        return num_tags, num_images

    def load_latents(self, image_hash, file_path, tag_name):
        # load vae and clip vectors
        bucket_name, input_file_path = separate_bucket_and_file_path(file_path)
        file_path = os.path.splitext(input_file_path)[0]

        clip_path = file_path + "_clip_h.msgpack"
        features_data = minio_manager.get_file_from_minio(self.minio_client, bucket_name, clip_path)
        clip_vector = msgpack.unpackb(features_data.data)["clip-feature-vector"]
        clip_vector = torch.tensor(clip_vector).squeeze()
        
        vae_latent_extension= f"_vae_latent.msgpack" if self.image_resolution == 512 else f"_vae_latent_{self.image_resolution}.msgpack"
        latent_path = file_path + vae_latent_extension
        latent_data = minio_manager.get_file_from_minio(self.minio_client, bucket_name, latent_path) 
        vae_latent = msgpack.unpackb(latent_data.data)["latent_vector"]
        vae_latent= torch.tensor(vae_latent).squeeze()

        return image_hash, tag_name, clip_vector, vae_latent
    
    def get_random_pseudo_epoch(self, epoch_size, score_threshold=0.6, random_seed=42):
        random.seed(random_seed)
        
        sampled_hashes, tags, file_paths = [], [], []
        # filter already used images
        filtered_images = [image for image in self.unique_images if image["image_hash"] not in self.used_image_hashes]

        random.shuffle(filtered_images)  # Shuffle images for random sampling
        sampled_images = filtered_images[:epoch_size]
        self.used_image_hashes.update([score["image_hash"] for score in sampled_images])

        for image in sampled_images:
            sampled_hashes.append(image["image_hash"])
            file_paths.append(image["file_path"])
            tags.append(image["tag"])
        
        return sampled_hashes, file_paths, tags

    def get_pseudo_epoch(self, epoch_size, score_threshold=0.6, random_seed=42):
        random.seed(random_seed)

        sampled_hashes, tags, file_paths = [], [], []
        tag_image_counts, filtered_tag_scores = {}, {}
        total_available_images = 0

        # Prepare filtered data for each tag
        for tag_name, scores in self.tag_scores.items():
            filtered_scores = [score for score in scores if score["image_hash"] not in self.used_image_hashes]
            count = len(filtered_scores)
            if count > 0:
                filtered_tag_scores[tag_name] = filtered_scores
                tag_image_counts[tag_name] = count
                total_available_images += count

        if total_available_images == 0:
            print("No images available after filtering and removing used images.")
            return [], [], []

        # Proportional sampling
        tag_sampling_counts = {tag: count / total_available_images for tag, count in tag_image_counts.items()}
        tag_samples = {tag: int(proportion * epoch_size) for tag, proportion in tag_sampling_counts.items()}
        remaining_samples = epoch_size - sum(tag_samples.values())

        # Adjust sampling counts if necessary
        tags_by_capacity = sorted(tag_image_counts.items(), key=lambda x: x[1], reverse=True)
        for tag_name, capacity in tags_by_capacity:
            if remaining_samples <= 0:
                break
            if tag_samples[tag_name] < capacity:
                tag_samples[tag_name] += 1
                remaining_samples -= 1

        # Sampling for each tag
        for tag_name, scores in filtered_tag_scores.items():
            num_to_sample = tag_samples[tag_name]
            if num_to_sample == 0:
                continue

            random.shuffle(scores)  # Shuffle scores for random sampling
            sampled_scores = scores[:num_to_sample]
            self.used_image_hashes.update([score["image_hash"] for score in sampled_scores])

            for score in sampled_scores:
                sampled_hashes.append(score["image_hash"])
                file_paths.append(score["file_path"])
                tags.append(tag_name)

        return sampled_hashes, file_paths, tags


    # # Sample filtered tag scores for the pseudo-epoch
    # def get_pseudo_epoch(self, epoch_size, score_threshold=0.6, random_seed=42):
    #     random.seed(random_seed)

    #     sampled_hashes, tags, image_clip_vectors, image_vae_latents = [], [], [], []
    #     tag_image_counts, filtered_tag_scores = {}, {}
    #     total_available_images = 0

    #     # Prepare filtered data for each tag
    #     for tag_name, scores in self.tag_scores.items():
    #         filtered_scores = [score for score in scores if score["image_hash"] not in self.used_image_hashes]
    #         count = len(filtered_scores)
    #         if count > 0:
    #             filtered_tag_scores[tag_name] = filtered_scores
    #             tag_image_counts[tag_name] = count
    #             total_available_images += count

    #     if total_available_images == 0:
    #         print("No images available after filtering and removing used images.")
    #         return [], [], [], []

    #     tag_sampling_counts = {tag: count / total_available_images for tag, count in tag_image_counts.items()}
    #     tag_samples = {tag: int(proportion * epoch_size) for tag, proportion in tag_sampling_counts.items()}
    #     remaining_samples = epoch_size - sum(tag_samples.values())

    #     # Adjust sampling counts if necessary
    #     tags_by_capacity = sorted(tag_image_counts.items(), key=lambda x: x[1], reverse=True)
    #     for tag_name, capacity in tags_by_capacity:
    #         if remaining_samples <= 0:
    #             break
    #         if tag_samples[tag_name] < capacity:
    #             tag_samples[tag_name] += 1
    #             remaining_samples -= 1

    #     # Load image latents on demand and sample images for the pseudo-epoch
    #     with ThreadPoolExecutor(max_workers=32) as executor:
    #         futures = []
    #         for tag_name, scores in filtered_tag_scores.items():
    #             num_to_sample = tag_samples[tag_name]
    #             if num_to_sample == 0:
    #                 continue

    #             random.shuffle(scores)  # Shuffle scores for random sampling
    #             sampled_scores = scores[:num_to_sample]
    #             self.used_image_hashes.update([score["image_hash"] for score in sampled_scores])

    #             # Submit latent loading tasks
    #             for score in sampled_scores:
    #                 futures.append(executor.submit(self.load_latents, score["image_hash"], score["file_path"], tag_name))

    #         # Collect results
    #         results = []
    #         for future in tqdm(as_completed(futures), total=len(futures)):
    #             try:
    #                 result = future.result()
    #                 results.append(result)
    #             except Exception as e:
    #                 print(f"Failed to load latents for an image: {e}")

    #     results.sort(key=lambda x: x[0])
    #     for image_hash, tag_name, clip_vector, vae_latent in results:
    #         sampled_hashes.append(image_hash)
    #         tags.append(tag_name)
    #         image_clip_vectors.append(clip_vector)
    #         image_vae_latents.append(vae_latent)

    #     return sampled_hashes, tags, image_clip_vectors, image_vae_latents

 
def parse_args():
    parser = argparse.ArgumentParser()

    # MinIO credentials
    parser.add_argument('--minio-access-key', type=str, required=True, help='Access key for MinIO storage.')
    parser.add_argument('--minio-secret-key', type=str, required=True, help='Secret key for MinIO storage.')
    parser.add_argument('--model-type', type=str, help='Model type elm-v1 or linear.', default="elm")

    return parser.parse_args()

def main():
    args= parse_args()

    # get minio client
    minio_client = minio_manager.get_minio_client(minio_access_key=args.minio_access_key,
                                        minio_secret_key=args.minio_secret_key)

    # initialize sampler
    proportional_sampler = ProportionalSamplingLoader(minio_client,
                                                   model_type= args.model_type)

    # create the sampling dataset
    proportional_sampler.create_sampling_datasets()

if __name__=="__main__":
    main()