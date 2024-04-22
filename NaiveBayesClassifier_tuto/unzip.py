
# ##########################
def unzip(file_name,foler_name):

    """unzip a file, file_name == input a string
    folder_name == input a string

    """
    global pwd_new
    import zipfile
    import os
    pwd = os.getcwd()
    # if user want to make a dir in the current working directory
    if foler_name:
        os.makedirs(foler_name, exist_ok=True)
        pwd_new= pwd + '/' + foler_name
    with zipfile.ZipFile(pwd + '/' + file_name, 'r') as zip_ref:
        zip_ref.extractall(pwd_new)
    return "File unzipped and saved in the folder 'Data' successfully!"
# ##########################
