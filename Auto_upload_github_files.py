import os
import shutil
import zipfile
import paramiko

# ========== CONFIGURABLE VARIABLES ==========
ZIP_PATH = r"C:/user/s.hun.lee/downloads/SimulGen-VAE-main.zip"
UNZIP_DIR = r"C:/user/s.hun.lee/downloads/SimulGen-VAE-main"
LOCAL_TARGET_DIR = r"D:/AI_projects/PCB_slit/ANN2"
REMOTE_HOST = "202.20.185.100"
REMOTE_PORT = 22
REMOTE_USER = "s.hun.lee"
REMOTE_PASS = "atleast12!"
REMOTE_TARGET_DIR = "/home/sr5/s.hun.lee/ML_ev/SimulGen_VAE/v2/PCB_slit/484_dataset/github"
# ============================================

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
    for root, dirs, files in os.walk(local_dir):
        rel_path = os.path.relpath(root, local_dir)
        remote_path = os.path.join(remote_dir, rel_path).replace('\\', '/')
        try:
            sftp.stat(remote_path)
        except FileNotFoundError:
            sftp.mkdir(remote_path)
        for file in files:
            local_file = os.path.join(root, file)
            remote_file = os.path.join(remote_path, file).replace('\\', '/')
            sftp.put(local_file, remote_file)

def main():
    print(f"Unzipping {ZIP_PATH} to {UNZIP_DIR}...")
    unzip_file(ZIP_PATH, UNZIP_DIR)
    print(f"Copying files from {UNZIP_DIR} to {LOCAL_TARGET_DIR}...")
    copy_all(UNZIP_DIR, LOCAL_TARGET_DIR)
    print(f"Uploading {LOCAL_TARGET_DIR} to {REMOTE_HOST}:{REMOTE_TARGET_DIR}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASS)
    sftp = ssh.open_sftp()
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
                sftp.mkdir(path)
    sftp_upload_dir(sftp, LOCAL_TARGET_DIR, REMOTE_TARGET_DIR)
    sftp.close()
    ssh.close()
    print("Done.")

if __name__ == "__main__":
    main()
