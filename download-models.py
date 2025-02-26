import gdown
import argparse
import os


def download_filtered_google_drive_folder(folder_url, destination_folder, prefix="horse"):
    """Downloads only the folders starting with a specific prefix from a Google Drive folder using gdown."""
    
    # List files and folders in the Google Drive folder
    file_list = gdown.download_folder(folder_url, output=destination_folder, quiet=True, remaining_ok=True, use_cookies=False)

    if not file_list:
        print("No files or folders found in the specified Google Drive folder.")
        return
    
    # Filter only the folders that start with the given prefix
    filtered_files = [f for f in file_list if os.path.basename(f).startswith(prefix)]
    
    if not filtered_files:
        print(f"No folders starting with '{prefix}' found.")
        return
    
    print(f"Downloading folders starting with '{prefix}':")
    for file in filtered_files:
        print(f"- {file}")
    
    # Download only the filtered files
    for file in filtered_files:
        gdown.download(file, os.path.join(destination_folder, os.path.basename(file)), quiet=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download specific checkpoints from Google Drive.")
    parser.add_argument("gdrive_folder_path", type=str, help="Google Drive folder URL")
    args = parser.parse_args()

    destination_folder = "checkpoints"

    # Download the filtered folder
    download_filtered_google_drive_folder(args.gdrive_folder_path, destination_folder)


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
