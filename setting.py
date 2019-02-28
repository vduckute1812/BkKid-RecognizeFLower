import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Dir path of this file (setting.py)

NAME_LIST = {
    "Hoa Canh Buom": 'cos',
    "Hoa Hong Mon": "ant",
    "Hoa Mao Ga": "coc",
    "Hoa Sen": "lot",
    "Hoa Thien Dieu": "str",
    "Hoa Rum":'cal',
    "Hoa Van Tho":'mar',
    "Hoa But":'hib',
    "Hoa Cam Tu Cau":'hor',
    "Hoa Cuc Trang":'dai',
    "Dahlia": 'dah',
    "Ixora": 'ixo',
    "Lily": 'lil',
    "Rose": 'ros',
    "Sun": 'sun'
}
test_pic = "../test.jpg"
source_flowers_version_path = "./Hoa2"
output_flowers_version_path = "../Build/version-4/"
test_pic = "../test.jpg"
ORIGIN_PATH = os.path.join(BASE_DIR, "Build/Origin")
EDGE_PATH = os.path.join(BASE_DIR, "Build/Edge")
GRAY_PATH = os.path.join(BASE_DIR, "Build/Gray")
UPDATE_PATH = os.path.join(BASE_DIR, "UpdateGreyData")
UPDATE_PATH2 = os.path.join(BASE_DIR, "UpdateGreyData2")
OUT_TRUE_FILE_CLUSION = os.path.join(BASE_DIR, "TrueClusion")
OUT_FALSE_FILE_CLUSION = os.path.join(BASE_DIR,"FalseClusion")
GRAY_HU = os.path.join(BASE_DIR,"Gray_Hu")
BINARY_PATH = os.path.join(BASE_DIR, "Build/Binary")
COLOR_PATH = os.path.join(BASE_DIR,"Build/Color")
CENTROID_PATH = os.path.join(BASE_DIR,"Build/CentroidImg")
OUT_PATH    = os.path.join(BASE_DIR, "Build/Test2")
OUT_PATH2   = os.path.join(BASE_DIR, "Build/Test3")
OUT_PATH3   = os.path.join(BASE_DIR, "Build/Test4")
TEST_OUT_PATH = os.path.join(BASE_DIR, "Test")
