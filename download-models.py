
import gdown
import argparse
import time

def download_google_drive_folder(folder_url, destination_folder, max_retries=5):
    """Downloads a Google Drive folder using gdown with retries."""
    for attempt in range(max_retries):
        try:
            gdown.download_folder(folder_url, output=destination_folder, quiet=False, resume=True)
            print("Download successful!")
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)  # Wait before retrying
    print("Download failed after multiple attempts.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download checkpoints.")
    parser.add_argument("gdrive_folder_path", type=str, help="Checkpoints path")
    args = parser.parse_args()
    
    destination_folder = "checkpoints"
    download_google_drive_folder(args.gdrive_folder_path, destination_folder)

# Original script commented
# import gdown
# import argparse
# def download_google_drive_folder(folder_url, destination_folder):
#     """Downloads a Google Drive folder using gdown."""
#     # Download the entire folder
#     gdown.download_folder(folder_url, output=destination_folder, quiet=False)

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description="download chekpoints.")
#     parser.add_argument("gdrive_folder_path", type=str, help="Checkpoints path")
#     args = parser.parse_args()
#     folder_path = args.gdrive_folder_path

#     #print(f"Hello, {args.folder_path}!")
    


#     # Destination folder where files will be downloaded
#     destination_folder = "checkpoints"

#     # Download the folder
#     download_google_drive_folder(folder_path, destination_folder)



