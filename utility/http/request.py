from urllib.parse import urlencode
import requests

# SERVER_ADDRESS = 'http://103.20.60.90:8764'
SERVER_ADDRESS = 'http://192.168.3.1:8111'

def http_add_external_image(image_data):
    url = SERVER_ADDRESS + "/external-images/add-external-image"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None

    try:
        response = requests.post(url, json=image_data, headers=headers)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}: {str(response.content)}")
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

def http_get_external_image_list(dataset, size=None):
    endpoint_url= "/external-images/get-all-external-image-list?dataset={}".format(dataset)

    if size:
        endpoint_url+= f"&size={size}"

    url = SERVER_ADDRESS + endpoint_url
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data_json = response.json()
            return data_json['response']['data']

    except Exception as e:
        print('request exception ', e)

def http_get_extract_image_list(dataset, size=None):
    endpoint_url= "/extracts/get-all-extracts-list?dataset={}".format(dataset)

    if size:
        endpoint_url+= f"&size={size}"

    url = SERVER_ADDRESS + endpoint_url
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data_json = response.json()
            return data_json['response']['data']

    except Exception as e:
        print('request exception ', e)

# Get list of all dataset names for extracts
def http_get_extract_dataset_list():
    url = SERVER_ADDRESS + "/datasets/list-datasets-v1"
    response = None

    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_json = response.json()
            datasets = data_json['response']['datasets']

            datasets= [dataset for dataset in datasets if dataset["bucket_id"]==1]
            return datasets

    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

def http_get_classifier_scores(classifier_id: int, image_sources:str, batch_size:int):
    scores=[]
    
    limit=batch_size
    offset=0

    while True:
        endpoint_url= "/pseudotag-classifier-scores/list-images-by-scores-v5?classifier_id={}&limit={}&offset={}&random_sampling=false&image_sources={}".format(classifier_id, limit, offset, image_sources)

        url = SERVER_ADDRESS + endpoint_url
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                data_json = response.json()
                image_batch= data_json['response']['images']
                num_images= len(image_batch)

                if num_images>0: 
                    scores.extend(image_batch)
                else:
                    break

            else:
                break

        except Exception as e:
            print('request exception ', e)
            break

        offset += num_images
    
    return scores

def http_get_image_details_by_hashes(image_hashes: list, fields: list):
    url = SERVER_ADDRESS + "/extract-images/get-image-details-by-hashes"
    params = {
        "image_hashes": image_hashes,
        "fields": fields
    }
    response = None
    try:
        response = requests.get(url, params= params)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")
            print(f"Error: {response.content.decode('utf-8')}")
            return []
        return response.json()["response"]["images"]
    except Exception as e:
        print('request exception ', e)
        
    finally:
        if response:
            response.close()

    return None

def http_get_classifier_model_list():
    url = SERVER_ADDRESS + "/pseudotag-classifiers/list-classifiers"
    response = None
    try:
        response = requests.get(url)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")
            return []
        return response.json()["response"]["classifiers"]
    except Exception as e:
        print('request exception ', e)
        
    finally:
        if response:
            response.close()

    return None

def http_get_tag_list():
    url = SERVER_ADDRESS + "/tags/list-tag-definitions"
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_json = response.json()
            return data_json["response"]["tags"]

    except Exception as e:
        print('request exception ', e)

# Function to get random images data (hash, uuid, file paths) by bucket and dataset from the API
def http_get_random_image_data(bucket_name, dataset_name, size):
    # url = f"http://103.20.60.90:8764/all-images/list-images-with-random-sampling?bucket_name={bucket_name}&dataset_name={dataset_name}&limit={size}&time_unit=minutes&random_sampling=true"
    url = SERVER_ADDRESS + f"/all-images/list-images-with-random-sampling?bucket_name={bucket_name}&dataset_name={dataset_name}&limit={size}&time_unit=minutes&random_sampling=true"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            response_json = response.json()
            image_data = response_json.get("response", {}).get("images", [])
            complete_time = response_json.get("request_complete_time", {})
            print(f"Fetched {size} images, request complete time: {complete_time}")
            return [(item["image_hash"], item["uuid"], item["image_path"]) for item in image_data]
        else:
            response_json = response.json()
            error_string = response_json.get("request_error_string", {})
            complete_time = response_json.get("request_complete_time", {})
            print(f"Failed to fetch images, status code: {response.status_code}, error: {error_string}, time: {complete_time}")
            
    except Exception as e:
        print(f"Error fetching images: {e}")
    return []

# Function to get random images data (hash, uuid, file paths) by bucket and dataset from the API
def http_get_random_extracts(dataset_name, size):
    url = SERVER_ADDRESS + f"/extracts/get-all-extracts-list-v1?dataset_name={dataset_name}&size={size}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            response_json = response.json()
            image_data = response_json.get("response", {}).get("data", [])
            complete_time = response_json.get("request_complete_time", {})
            print(f"Fetched {size} images, request complete time: {complete_time}")
            return image_data
        else:
            response_json = response.json()
            error_string = response_json.get("request_error_string", {})
            complete_time = response_json.get("request_complete_time", {})
            print(f"Failed to fetch images, status code: {response.status_code}, error: {error_string}, time: {complete_time}")
            
    except Exception as e:
        print(f"Error fetching images: {e}")
    return []

# Function to get all image data (hash, uuid, file paths) by bucket and dataset from the API
def http_get_all_image_data(bucket_name, dataset_name, size):
    # url = f"http://103.20.60.90:8764/all-images/list-images-with-random-sampling?bucket_name={bucket_name}&dataset_name={dataset_name}&limit={size}&time_unit=minutes&random_sampling=false"
    url = SERVER_ADDRESS + f"/all-images/list-images-with-random-sampling?bucket_name={bucket_name}&dataset_name={dataset_name}&limit={size}&time_unit=minutes&random_sampling=false"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            response_json = response.json()
            image_data = response_json.get("response", {}).get("images", [])
            complete_time = response_json.get("request_complete_time", {})
            print(f"Fetched {size} images, request complete time: {complete_time}")
            return [(item["image_hash"], item["uuid"], item["image_path"]) for item in image_data]
        else:
            response_json = response.json()
            error_string = response_json.get("request_error_string", {})
            complete_time = response_json.get("request_complete_time", {})
            print(f"Failed to fetch images, status code: {response.status_code}, error: {error_string}, time: {complete_time}")
            
    except Exception as e:
        print(f"Error fetching images: {e}")
    return []

def http_get_images_in_batches(dataset_name: str, batch_size: int = 500000):
    images = []
    offset = 0

    while True:
        # Construct query parameters
        query_params = {
            "dataset": dataset_name,
            "limit": batch_size,
            "offset": offset
        }

        encoded_params = urlencode(query_params)  # Properly formats the query string
        url = f"{SERVER_ADDRESS}/extract-images/list-images-v1?{encoded_params}"
        
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                data_json = response.json()
                image_batch = data_json['response']['images']
                images.extend(image_batch)
                num_images = len(image_batch)
                print(f"loaded {num_images} images.")

                if num_images < batch_size:
                    break
            else:
                print(f"Error: Received status code {response.status_code}")
                break

        except Exception as e:
            print('Request exception:', e)
            break

        offset += num_images

    return images


# http://103.20.60.90:8764/pseudotag-classifier-scores/list-images-by-scores-with-tag-string?tag_string=game-3d&model_type=elm&min_score=0&max_score=1&limit=10&random_sampling=true&image_source=extract_image
def http_get_images_by_tag(tag_string, model_type="elm", min_score=0.7, max_score=None, limit=None, offset=None, random_sampling="true", image_source="extract_image"):
    # Construct the request URL. When limit is an empty string, the limit parameter is not included.
    url = SERVER_ADDRESS + f"/pseudotag-classifier-scores/list-images-by-scores-with-tag-string?tag_string={tag_string}&model_type={model_type}&min_score={min_score}&random_sampling={random_sampling}&image_source={image_source}"
    
    if limit:
        url += f"&limit={limit}"
    
    if offset:
        url += f"&offset={offset}"

    if max_score is not None:
        url = url.replace(f"&min_score={min_score}", f"&min_score={min_score}&max_score={max_score}")

    try:
        response = requests.get(url)
        if response.status_code == 200:
            response_json = response.json()
            image_data = response_json.get("response", {}).get("images", [])
            complete_time = response_json.get("request_complete_time", {})
            print(f"Fetched {len(image_data)} images, request complete time: {complete_time}")
            
            # Check for missing fields and log the problematic image data
            valid_data = []
            for item in image_data:
                image_hash = item.get("image_hash")
                score = item.get("score")
                tag_id = item.get("tag_id")
                image_uuid = item.get("image_uuid")
                image_path = item.get("file_path")

                # if not image_uuid or not image_path:
                #     print(f"Missing required data for image: Hash: {image_hash}, Score: {score}, Tag ID: {tag_id}, UUID: {image_uuid}, Path: {image_path}")
                # else:
                #     valid_data.append((image_hash, score, tag_id, image_uuid, image_path))

                valid_data.append((image_hash, score, tag_id, image_uuid, image_path))
            return valid_data  # Return only valid data

        else:
            response_json = response.json()
            error_string = response_json.get("request_error_string", {})
            complete_time = response_json.get("request_complete_time", {})
            print(f"Failed to fetch images, status code: {response.status_code}, error: {error_string}, time: {complete_time}")
            
    except Exception as e:
        print(f"Error fetching images: {e}")
    
    return []

# get classifier 
def http_get_images_by_tag_by_batches(tag_string, model_type="elm", min_score=0.6, max_score=None, image_source="extract_image", batch_size=300000):
    scores=[]

    limit=batch_size
    offset=0

    while True:
        image_batch = []

        try:
            url = SERVER_ADDRESS + f"/pseudotag-classifier-scores/list-images-by-scores-with-tag-string?tag_string={tag_string}&model_type={model_type}&min_score={min_score}&random_sampling=false&image_source={image_source}"

            if limit:
                url += f"&limit={limit}"

            if offset:
                url += f"&offset={offset}"

            if max_score:
                url += f"&max_score={max_score}"

            response = requests.get(url)
            if response.status_code == 200:
                response_json = response.json()
                image_batch = response_json.get("response", {}).get("images", [])
                complete_time = response_json.get("request_complete_time", {})
                print(f"Fetched {len(image_batch)} images, request complete time: {complete_time}")
            else:
                image_batch=[]
    
            num_images= len(image_batch)
            # add the score batch to all scores  
            scores.extend(image_batch)

            if num_images<batch_size:
                break

        except Exception as e:
            print('request exception ', e)
            break

        offset += num_images
        print(f"loaded {offset} scores")
    
    return scores

# get list of rank sigma-scores images
def http_get_list_rank_scores_with_rank_string(rank_string, model_type, limit, min_sigma_score, random_sampling=False, image_source="extract_image"):
    url = SERVER_ADDRESS + "/image-scores/scores/list-rank-scores-with-rank-string"
    params = {
        "rank_string": rank_string,
        "model_type": model_type,
        "limit": limit,
        "min_score": min_sigma_score,
        "random_sampling": str(random_sampling).lower(),
        "image_source": image_source,
    }
    response = None

    try:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data_json = response.json()
            # print(f"Response JSON: {data_json}")

            # Safely access nested keys
            scores = data_json.get('response', {}).get('scores')
            if scores is not None:
                return scores
            else:
                print("The 'scores' key is missing in the response.")
                return None
        else:
            print(f"Request failed with status code: {response.status_code}")
            return None

    except Exception as e:
        print('Request exception:', e)
        return None

    finally:
        if response:
            response.close()

# http://103.20.60.90:8764/tags/list-tag-definitions
def fetch_tag_strings():
    url = SERVER_ADDRESS + f"/tags/list-tag-definitions"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()  
            tags = data.get("response", {}).get("tags", [])  
            tag_strings = [tag.get("tag_string") for tag in tags]
            return tag_strings
        else:
            print(f"Failed to fetch tags, status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching tags: {str(e)}")
        return []


# http://103.20.60.90:8764/extract-images/get-image-details-by-hash/2f14d74d730b4b21a5e0daa48cb4312f9f1f9fbc3321b337cb0d2f1c0c98211b
def http_get_image_path_by_hash(image_hash):
    url = SERVER_ADDRESS + f"/extract-images/get-image-details-by-hash/{image_hash}"
    try:
        response = requests.get(url, headers={'accept': 'application/json'})
        if response.status_code == 200:
            response_json = response.json()
            file_path = response_json.get("response", {}).get("file_path", "")
            print(f"Fetched image path for hash {image_hash}: {file_path}")
            return file_path
        else:
            response_json = response.json()
            error_string = response_json.get("request_error_string", {})
            print(f"Failed to fetch image path by hash, status code: {response.status_code}, error: {error_string}")
    
    except Exception as e:
        print(f"Error fetching image path: {e}")
    
    return None


def http_get_external_dataset_in_batches_with_extracts(dataset: str, batch_size: int):
    external_images=[]
    
    limit=batch_size
    offset=0

    while True:
        endpoint_url= "/external-images/list-images-with-extracts?dataset={}&limit={}&offset={}&order=asc".format(dataset, limit, offset)

        url = SERVER_ADDRESS + endpoint_url
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                data_json = response.json()
                image_batch= data_json['response']['images']
                num_images= len(image_batch)

                if num_images>0: 
                    external_images.extend(image_batch)
                else:
                    break

            else:
                break

        except Exception as e:
            print('request exception ', e)
            break

        offset += num_images

        print(f"Loaded {offset} images")
    
    return external_images

def http_get_tagged_images_by_image_type(tag_id, image_type = "all_resolutions"):
    url = SERVER_ADDRESS + "/tags/get-images-by-image-type/?tag_id={}&image_type={}".format(tag_id, image_type)
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_json = response.json()
            return data_json["response"]["images"]

    except Exception as e:
        print('request exception ', e)

def http_add_classifier_model(model_card):
    url = SERVER_ADDRESS + "/pseudotag-classifiers/register-tag-classifier"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None

    try:
        response = requests.post(url, data=model_card, headers=headers)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")
        print("classifier data=", response.content)
        return response.content
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

def http_add_ranking_model(model_data):
    url = SERVER_ADDRESS + "/ranking-models/register-ranking-model"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None

    try:
        response = requests.post(url, data=model_data, headers=headers)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")
        print("ranking model data=", response.content)
        return response.content
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

def http_get_rank_list():
    url = SERVER_ADDRESS +  "/ab-rank/list-ranks"   # old deprecated "/ab-rank/list-rank-models"
    response = None
    try:
        response = requests.get(url)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")
            return []
        return response.json()["response"]["ranks"]
    except Exception as e:
        print('request exception ', e)
        
    finally:
        if response:
            response.close()

    return None