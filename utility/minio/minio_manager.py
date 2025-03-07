import io
import time
from minio import Minio
import os
import requests
from tqdm import tqdm
from .progress import Progress
from utility.utils_logger import logger

# TODO: rename file to minio_manager
# TODO: remove hardcode in the future
#  use config file

MINIO_ADDRESS = "192.168.3.6:9000"


def get_minio_client(minio_access_key, minio_secret_key, minio_ip_addr=None):
    global MINIO_ADDRESS

    if minio_ip_addr is not None:
        MINIO_ADDRESS = minio_ip_addr
    # check first if minio client is available
    minio_client = None
    # there should be a timer between connection attempts
    while minio_client is None:
        # check minio server
        if is_minio_server_accessible(MINIO_ADDRESS):
            minio_client = connect_to_minio_client(MINIO_ADDRESS, minio_access_key, minio_secret_key)
            return minio_client
        else:
            # sleep for 0.1 seconds before re-attempt
            print("WARNING: utility.minio, get_minio_client, Minio server accessibility check failed.")
            time.sleep(0.1)


def connect_to_minio_client(minio_ip_addr=None, access_key=None, secret_key=None,):
    global MINIO_ADDRESS

    if minio_ip_addr is not None:
        MINIO_ADDRESS = minio_ip_addr

    print("Connecting to minio client...")
    # TODO: add an exception
    client = Minio(MINIO_ADDRESS, access_key, secret_key, secure=False)
    print("Successfully connected to minio client...")
    return client


def is_minio_server_accessible(address=None):
    if address is None:
        address = MINIO_ADDRESS
    
    print("Checking if minio server is accessible...")
    try:
        r = requests.head("http://" + address + "/minio/health/live", timeout=5)
    except requests.Timeout as e:
        print(f"WARNING: utility.minio, is_minio_server_accessible, Minio server keep alive request timeout error, Error: {e}")
    except ConnectionError as e:
        print(f"WARNING: utility.minio, is_minio_server_accessible, Connection error, Error: {e}")
    except requests.HTTPError as e:
        print(f"WARNING: utility.minio, is_minio_server_accessible, Http error, Error: {e}")
    except requests.RequestException as e:
        print(f"WARNING: utility.minio, is_minio_server_accessible, Request exception error, Error: {e}")
    except Exception as e:
        print(f"WARNING: utility.minio, is_minio_server_accessible, Error: {e}")
    
    if r.status_code != 200:
        print(f"WARNING: utility.minio, is_minio_server_accessible, r.status_code!=200, status code: {r.status_code}")

    # if status code != 200, print the status code, and put the function name in the error print
    return r.status_code == 200

def download_from_minio(client, bucket_name, object_name, output_path):
    if not os.path.isfile(output_path):
        client.fget_object(bucket_name, object_name, output_path, progress=Progress())
    else:
        logger.info(f"{object_name} already exists.")

def download_folder_from_minio(client, bucket_name, folder_name, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # List all objects in the Minio folder
    objects = client.list_objects(bucket_name, prefix=folder_name, recursive=True)
    
    for obj in objects:
        object_name = obj.object_name
        relative_path = os.path.relpath(object_name, folder_name)
        output_path = os.path.join(output_folder, relative_path)
        
        if not os.path.isfile(output_path):
            # Create directories if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Download the object
            client.fget_object(bucket_name, object_name, output_path)
        else:
            logger.info(f"{object_name} already exists.")


def download_dataset_from_minio(client, bucket_name, folder_name):
       
    # List all objects in the Minio folder
    objects = client.list_objects(bucket_name, prefix=folder_name, recursive=True)

    # list of the data which is downloaded
    data = []
    for obj in objects:
        object_name = obj.object_name
        
        # Download the object
        data.append[client.get_object(bucket_name, object_name, progress=Progress())]
        
    return data
        

def get_file_from_minio(client, bucket_name, file_name):
    try:
        # Get object data
        data = client.get_object(bucket_name, file_name)

        return data

    except Exception as err:
        print(f"Error: {err}")

    return None

def get_list_of_buckets(client):
    buckets = client.list_buckets()
    for bucket in buckets:
        print("Bucket: {0}: {1}".format(bucket.name, bucket.creation_date))

    return buckets


def check_if_bucket_exists(client, bucket_name):
    if client.bucket_exists(bucket_name):
        print("{0} exists".format(bucket_name))
        return True

    print("{0} does not exists".format(bucket_name))
    return False


def create_bucket(client, bucket_name):
    client.make_bucket(bucket_name)
    print("Bucket: {0} successfully created...".format(bucket_name))


def remove_bucket(client, bucket_name):
    client.remove_bucket(bucket_name)
    print("Bucket: {0} successfully deleted...".format(bucket_name))


def get_list_of_objects(client, bucket_name):
    object_names = []
    objects = client.list_objects(bucket_name)

    for obj in objects:
        obj_name = obj.object_name.replace('/', '')
        object_names.append(obj_name)

    return object_names



def get_list_of_objects_with_prefix(client, bucket_name, prefix):
    object_names = []
    objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)

    for obj in objects:
        object_names.append(obj.object_name)

    return object_names


def upload_from_file(client, bucket_name, object_name, file_path):
    result = client.fput_object(bucket_name, object_name, file_path)
    print(
        "created {0} object; etag: {1}, version-id: {2}".format(
            result.object_name, result.etag, result.version_id,
        ),
    )


def upload_data(client, bucket_name, object_name, data):
    try:
        result = client.put_object(
            bucket_name, object_name, data, length=-1, part_size=10 * 1024 * 1024,
        )
        print(
            "created {0} object; etag: {1}, version-id: {2}".format(
                result.object_name, result.etag, result.version_id,
            ),
        )

    except Exception as e:
        raise Exception(e)

def upload_data_progress(client, bucket_name, object_name, data):
    try:
        # Wrap the data stream with tqdm for progress reporting
        # Ensure 'data' is a file-like object with a 'tell' method to work correctly
        if isinstance(data, bytes):
            data = io.BytesIO(data)
        
        total_size = data.getbuffer().nbytes if isinstance(data, io.BytesIO) else None
        tqdm_stream = tqdm(total=total_size, unit='B', unit_scale=True, desc=object_name)

        # Define a custom class to wrap the data stream and update the progress bar
        class TqdmUpdater(io.BufferedReader):
            def read(self, size=-1):
                chunk = super().read(size)
                tqdm_stream.update(len(chunk))
                return chunk

        wrapped_data = TqdmUpdater(data)

        # Perform the upload with the wrapped data stream
        result = client.put_object(
            bucket_name, object_name, wrapped_data, length=total_size, part_size=10 * 1024 * 1024,
        )
        tqdm_stream.close()
        print(
            "created {0} object; etag: {1}, version-id: {2}".format(
                result.object_name, result.etag, result.version_id,
            ),
        )

    except Exception as e:
        tqdm_stream.close()  # Ensure closure of tqdm on error
        raise Exception(e)


def remove_an_object(client, bucket_name, object_name):
    # Remove object.
    client.remove_object(bucket_name, object_name)


def is_object_exists(client, bucket_name, object_name):
    try:
        result = client.stat_object(bucket_name, object_name)
        # print(
        #     "last-modified: {0}, size: {1}".format(
        #         result.last_modified, result.size,
        #     ),
        # )

        if result.object_name != "":
            return True
    except Exception as e:
        return False

    return False
