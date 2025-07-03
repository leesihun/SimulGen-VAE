import os
import shutil
import zipfile
import paramiko
import requests

# ========== CONFIGURABLE VARIABLES ==========
GITHUB_ZIP_URL = "https://github.com/leesihun/SimulGen-VAE/archive/refs/heads/main.zip"
ZIP_PATH = r"C:/Users/s.hun.lee/Downloads/SimulGen-VAE-main.zip"
UNZIP_DIR = r"C:/Users/s.hun.lee/Downloads/SimulGen-VAE-main"
LOCAL_TARGET_DIR = r"D:/AI_projects/PCB_slit/ANN2"
REMOTE_HOST = "202.20.185.100"
REMOTE_PORT = 22
REMOTE_USER = "s.hun.lee"
REMOTE_PASS = "atleast12!"
REMOTE_TARGET_DIR = "/home/sr5/s.hun.lee/ML_ev/SimulGen_VAE/v2/PCB_slit/484_dataset/github"
# ============================================

def download_github_zip(url, dest_path):
    print(f"Downloading {url} ...")
    try:
        response = requests.get(url, verify=r"C:/DigitalCity.crt")
    except requests.exceptions.SSLError as e:
        print(f"SSL error: {e}")
        return
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded to {dest_path}")
    else:
        print(f"Failed to download: {response.status_code}")

def unzip_file(zip_path, extract_to):
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(extract_to))
    # The zip will extract to a folder, so we rename/move it to extract_to
    extracted_folder = os.path.join(os.path.dirname(extract_to), os.path.basename(zip_path).replace('.zip', ''))
    if os.path.exists(extracted_folder) and extracted_folder != extract_to:
        shutil.move(extracted_folder, extract_to)

def copy_all(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def sftp_upload_dir(sftp, local_dir, remote_dir):
    # Recursively upload a directory to the remote server, overwriting files
    total_files = 0
    success_files = 0
    failed_files = 0
    for root, dirs, files in os.walk(local_dir):
        rel_path = os.path.relpath(root, local_dir)
        remote_path = os.path.join(remote_dir, rel_path).replace('\\', '/')
        try:
            sftp.stat(remote_path)
        except FileNotFoundError:
            sftp.mkdir(remote_path)
        for file in files:
            total_files += 1
            local_file = os.path.join(root, file)
            remote_file = os.path.join(remote_path, file).replace('\\', '/')
            try:
                sftp.put(local_file, remote_file)
                # Verify file size
                local_size = os.path.getsize(local_file)
                remote_size = sftp.stat(remote_file).st_size
                if local_size == remote_size:
                    print(f"Uploaded and verified: {remote_file}")
                    success_files += 1
                else:
                    print(f"WARNING: Size mismatch for {remote_file} (local: {local_size}, remote: {remote_size})")
                    failed_files += 1
            except Exception as e:
                print(f"Failed to upload {local_file} to {remote_file}: {e}")
                failed_files += 1
    print(f"Upload summary: {success_files}/{total_files} files succeeded, {failed_files} failed.")

def main():
    print(f"Downloading {GITHUB_ZIP_URL} to {ZIP_PATH}...")
    download_github_zip(GITHUB_ZIP_URL, ZIP_PATH)
    print(f"Unzipping {ZIP_PATH} to {UNZIP_DIR}...")
    unzip_file(ZIP_PATH, UNZIP_DIR)
    print(f"Copying files from {UNZIP_DIR} to {LOCAL_TARGET_DIR}...")
    copy_all(UNZIP_DIR, LOCAL_TARGET_DIR)
    print(f"Uploading {LOCAL_TARGET_DIR} to {REMOTE_HOST}:{REMOTE_TARGET_DIR}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASS, timeout=10)
        print("SSH connection established.")
    except paramiko.AuthenticationException:
        print("Authentication failed, please verify your credentials.")
        return
    except paramiko.SSHException as sshException:
        print(f"Unable to establish SSH connection: {sshException}")
        return
    except Exception as e:
        print(f"Exception in connecting to the server: {e}")
        return
    try:
        sftp = ssh.open_sftp()
        print("SFTP session established.")
    except Exception as e:
        print(f"Failed to open SFTP session: {e}")
        ssh.close()
        return
    # Ensure remote target dir exists
    try:
        sftp.stat(REMOTE_TARGET_DIR)
    except FileNotFoundError:
        # Recursively create directories
        dirs = REMOTE_TARGET_DIR.strip('/').split('/')
        path = ''
        for d in dirs:
            path += '/' + d
            try:
                sftp.stat(path)
            except FileNotFoundError:
                try:
                    sftp.mkdir(path)
                    print(f"Created remote directory: {path}")
                except Exception as e:
                    print(f"Failed to create remote directory {path}: {e}")
                    sftp.close()
                    ssh.close()
                    return
    try:
        sftp_upload_dir(sftp, LOCAL_TARGET_DIR, REMOTE_TARGET_DIR)
        print("Upload completed.")
    except Exception as e:
        print(f"Error during file upload: {e}")
        sftp.close()
        ssh.close()
        return
    # Verification: List remote files
    try:
        print("Remote directory contents after upload:")
        for entry in sftp.listdir_attr(REMOTE_TARGET_DIR):
            print(f"  {entry.filename}")
    except Exception as e:
        print(f"Could not list remote directory: {e}")
    sftp.close()
    ssh.close()
    # Delete the ZIP file
    try:
        os.remove(ZIP_PATH)
        print(f"Deleted ZIP file: {ZIP_PATH}")
    except Exception as e:
        print(f"Could not delete ZIP file: {e}")
    # Delete the unzipped folder
    try:
        shutil.rmtree(UNZIP_DIR)
        print(f"Deleted unzipped folder: {UNZIP_DIR}")
    except Exception as e:
        print(f"Could not delete unzipped folder: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
