# -*- coding: utf-8 -*-
"""

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author: Erik A. Kneller
"""
import zipfile
import os


def check_for_zipped_zmap(direc_path, file_name):
    file_path_zip = os.path.join(direc_path, file_name + '.zip')
    ifind_zip = 0
    if os.path.isfile(file_path_zip) == True:
        ifind_zip = 1
    return ifind_zip
    
    
def unzip_zmap(direc_path, file_name):
    file_path_zip = os.path.join(direc_path, file_name + '.zip')
    with zipfile.ZipFile(file_path_zip, 'r') as zip_ref:
        zip_ref.extractall(direc_path)


def zip_zmap(icompress, direc_path, file_name):
    if icompress == 1:
        compression = zipfile.ZIP_DEFLATED
        zf = zipfile.ZipFile(
                                os.path.join(direc_path, file_name + '.zip'), 
                                mode='w')
        try:
            zf.write(
                        os.path.join(direc_path, file_name), 
                        file_name, compress_type=compression)
            zf.close()    
            os.remove(direc_path + file_name)
        except:
            zf.close()
        