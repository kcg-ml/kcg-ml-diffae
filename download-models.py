import gdown
import argparse
def download_google_drive_folder(folder_url, destination_folder):
    """Downloads a Google Drive folder using gdown."""
    # Download the entire folder
    gdown.download_folder(folder_url, output=destination_folder, quiet=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="download chekpoints.")
    parser.add_argument("gdrive_folder_path", type=str, help="Checkpoints path")
    args = parser.parse_args()
    folder_path = args.gdrive_folder_path

    #print(f"Hello, {args.folder_path}!")
    


    # Destination folder where files will be downloaded
    destination_folder = "checkpoints"

    # Download the folder
    download_google_drive_folder(folder_path, destination_folder)
