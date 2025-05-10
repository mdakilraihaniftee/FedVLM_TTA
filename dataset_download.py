import os
import requests
import tarfile
import zipfile
from tqdm import tqdm

def download_only(url, save_dir, filename=None):
    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = os.path.basename(url.split("?")[0])
    file_path = os.path.join(save_dir, filename)

    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=filename, ncols=100
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"Downloaded file saved as: {file_path}")
    else:
        raise Exception(f"Failed to download file. HTTP status code: {response.status_code}")

    return file_path


def download_and_extract_tar(url, save_dir, filename=None):
    """
    Downloads and extracts a .tar, .tar.gz, or .zip file.

    Args:
        url (str): Direct download link to the archive.
        save_dir (str): Directory to save and extract the file.
        filename (str, optional): Name for the downloaded file.
    """
    archive_path = download_only(url, save_dir, filename)

    print("Extracting...")
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(save_dir)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    print(f"Extracted contents to: {save_dir}")


download_DTD_flag = True
if download_DTD_flag:
    download_and_extract_tar(
    url="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
    save_dir="dataset"
    )
    download_only(
        url = "https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x", 
        save_dir="dataset/dtd",
        filename="split_zhou_DescribableTextures.json"
    )


download_EuroSAT_flag =  False
if download_EuroSAT_flag:
    pass
    download_and_extract_tar(
    url="https://madm.dfki.de/files/sentinel/EuroSAT.zip",
    save_dir="dataset/eurosat"
    )
    download_only(
        url = "https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o", 
        save_dir="dataset/eurosat",
        filename="split_zhou_EuroSAT.json"
    )

download_caltech_101_flag =  True
if download_caltech_101_flag:
    pass
    download_and_extract_tar(
    url="http://www.vision.caltech.edu/visipedia-data/Caltech101/Caltech101.tar.gz",
    save_dir="dataset/caltech-101"
    )
    download_only(
        url = "https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN", 
        save_dir="dataset/caltech-101",
        filename="split_zhou_Caltech101.json"
    )

download_Flower_102_flag =  True
if download_Flower_102_flag:
    download_and_extract_tar(
    url="http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
    save_dir="dataset/oxford_flowers"
    )
    download_only(
    url="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
    save_dir="dataset/oxford_flowers", 
    filename="imagelabels.mat"
    )
    download_only(
        url = "https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT", 
        save_dir="dataset/oxford_flowers",
        filename="split_zhou_OxfordFlowers.json"
    )
    download_only(
        url = "https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0", 
        save_dir="dataset/oxford_flowers",
        filename="cat_to_name.json"
    )

download_Food_101_flag =  True
if download_Food_101_flag:
    download_and_extract_tar(
    url="https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
    save_dir="dataset"
    )
    download_only(
        url = "https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl", 
        save_dir="dataset/food-101",
        filename="split_zhou_Food101.json"
    )
    

download_OxfordPets_flag = True
if download_OxfordPets_flag:
    download_and_extract_tar(
    url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    save_dir="dataset/oxford_pets"
    )
    download_and_extract_tar(
    url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
    save_dir="dataset/oxford_pets"
    )
    download_only(
        url = "https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN", 
        save_dir="dataset/oxford_pets",
        filename="split_zhou_OxfordPets.json"
    )
    


download_StanfordCars_flag = False
if download_StanfordCars_flag:
    pass
    download_and_extract_tar(
    url="http://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
    save_dir="dataset/stanford_cars"
    )
    download_and_extract_tar(
    url="http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
    save_dir="dataset/stanford_cars"
    )
    download_only(
        url = "https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT", 
        save_dir="dataset/stanford_cars",
        filename="split_zhou_StanfordCars.json"
    )

download_SUN397_flag =  False
if download_SUN397_flag:
    pass
    download_and_extract_tar(
    url="http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz",
    save_dir="dataset/sun397"
    )
    download_and_extract_tar(
    url="https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip",
    save_dir="dataset/sun397"
    )
    download_only(
        url = "https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq", 
        save_dir="dataset/sun397",
        filename="split_zhou_SUN397.json"
    )




download_UCF101_flag =  False
if download_UCF101_flag:
    pass
    download_and_extract_tar(
    url="https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O",
    save_dir="dataset/ucf101"
    )
    download_only(
        url = "https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y", 
        save_dir="dataset/ucf101",
        filename="split_zhou_UCF101.json"
    )


