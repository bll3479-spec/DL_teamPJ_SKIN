import os

# 폴더명을 바꿀 대상의 경로 가져오기
TS = os.listdir(r'./Data/Training/01_Source_Data')
TL = os.listdir(r'./Data/Training/02_Labeling_Data')
VS = os.listdir(r'./Data/Validation/01_Source_Data')
VL = os.listdir(r'./Data/Validation/02_Labeling_Data')

#한글 병명 -> 영문 병명 매핑 딕셔너리 생성
name_map = {
    '광선각화증' : 'actinic_keratosis',
    '기저세포암' : 'basal_cell_carcinoma',
    '멜라닌세포모반' : 'melanocytic_nevus',
    '보웬병' : 'bowen_disease',
    '비립종' : 'milia',
    '사마귀' : 'wart',
    '악성흑색종' : 'malignant_melanoma',
    '지루각화증' : 'seborrheic_keratosis',
    '편평세포암' : 'squamous_cell_carcinoma',
    '표피낭종' : 'epidermal_cyst',
    '피부섬유종' : 'dermatofibroma',
    '피지샘증식증' : 'sebaceous_hyperplasia', 
    '혈관종' : 'hemangioma',
    '화농_육아종' : 'pyogenic_granuloma',
    '흑색점' : 'lentigo'
}

#폴더 순회하며 접두사(TL_...)와 한글 병명 분리 및 
#name_map으로 영문 병명을 찾아서 변경

# 쪼개서 생각하기:
# #1단계: 폴더에서 접두사 / 한글병명 분리
# item = 'TL_광선각화증'
# parts = item.split('_', 1)
# #분리한 값을 각각 저장
# prefix = parts[0]
# korean = parts[1]
# #2딘계: 저장한 리스트를 이용하여 딕셔너리[키]로 영문병명 조회
# english = name_map[korean]

# #3단계:접두사와 영문병명 합치기
# #new_name = prefix + '_' + english
# new_name = f'{prefix}_{english}'
# print(new_name)

#4단계: 전체 폴더 순회하며 이름 변경점 확인

base_path = r'./Data/Training/02_Labeling_Data'

for item in TL:
    prefix, korean = item.split('_', 1)
    english = name_map[korean]
    new_name = f'{prefix}_{english}'

    old_path = os.path.join(base_path, item)
    new_path = os.path.join(base_path, new_name)
#5단계: 이름 바꾸기
    os.rename(old_path, new_path)
    print(old_path, '->', new_path)
