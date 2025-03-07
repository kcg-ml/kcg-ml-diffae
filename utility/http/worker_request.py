import json
import requests
#SERVER_ADDRESS = 'http://103.20.60.90:7764'
SERVER_ADDRESS = 'http://192.168.3.1:8211'

# Get request to get an available job
def http_get_job(worker_type: str = None, model_name: str = None):
    url = SERVER_ADDRESS + "/queue/model-training/get-job"
    response = None
    query_params = []

    if worker_type is not None:
        query_params.append("task_type={}".format(worker_type))
    if model_name is not None:
        query_params.append("model_name={}".format(model_name))

    if query_params:
        url += "?" + "&".join(query_params)

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

# Get request to get a completed job
def http_get_completed_job(model_id:str, model_name: str = None):
    url = SERVER_ADDRESS + "/queue/model-training/get-completed-job"
    response = None
    query_params = ["model_id={}".format(model_id)]

    if model_name is not None:
        query_params.append("model_name={}".format(model_name))

    if query_params:
        url += "?" + "&".join(query_params)

    try:
        response = requests.get(url)

        if response.status_code == 200:
            job_json = response.json()
            return job_json

    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

# post request to add a job to worker queue
def http_add_job(job):
    url = SERVER_ADDRESS + "/queue/model-training/add-job"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None
    
    try:
        response = requests.post(url, json=job, headers=headers)
        if response.status_code != 201 and response.status_code != 200:
            print(f"POST request failed with status code: {response.status_code}")

        decoded_response = json.loads(response.content.decode())
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return decoded_response

def http_update_job_completed(job):
    url = SERVER_ADDRESS + "/queue/model-training/update-completed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None

    try:
        response = requests.put(url, json=job, headers=headers)
        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")    
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

def http_update_job_failed(job):
    url = SERVER_ADDRESS + "/queue/model-training/update-failed"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None

    try:
        response = requests.put(url, json=job, headers=headers)
        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

# Get request to get sequential id
def http_get_model_sequential_id(model_id: str):
    url = SERVER_ADDRESS + f"/models/model-sequential-id?model_id={model_id}"
    response = None

    try:
        response = requests.get(url)
        if response.status_code == 200:
            job_json = response.json()
            return job_json["sequential_id"]
        
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

def http_add_model(model):
    url = SERVER_ADDRESS + "/models/add-model"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None

    try:
        response = requests.post(url, data=model, headers=headers)

        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")
        print("model_id=", response.json())
        return response.json()
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

    return None

def http_update_model_file_path(update_data):
    url = SERVER_ADDRESS + "/models/update-model-file-path"
    headers = {"Content-type": "application/json"}  # Setting content type header to indicate sending JSON data
    response = None

    try:
        response = requests.put(url, json=update_data, headers=headers)
        if response.status_code != 200:
            print(f"request failed with status code: {response.status_code}")
    except Exception as e:
        print('request exception ', e)

    finally:
        if response:
            response.close()

def http_get_model_details(sequence_number: int):
    url = SERVER_ADDRESS + "/models/get-model-details?sequence_number={}".format(sequence_number)
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_json = response.json()
            return data_json

    except Exception as e:
        print('request exception ', e)