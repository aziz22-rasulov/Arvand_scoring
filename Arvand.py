import numpy as np
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Заголовок
st.subheader('Добро пожаловать в скоринговую систему банка Arvand')

# Загрузка моделей и данных





with open("lr_model.pkl", "rb") as pickle_in:
    regression1 = joblib.load(pickle_in)

with open("model_gb.pkl", "rb") as pickle_in:
    regression2 = joblib.load(pickle_in)

# Создание DataFrame с данными
data1 = pd.DataFrame({'Married': ['beoila', 'oilador', 'bevamard', 'judoshuda']})
data2 = pd.DataFrame({'Nationality': ['Uzbek', 'Tojik', 'Totor', 'Rus', 'Kirgiz', 'Ukrain', 'Others', 'Karis', 'Karachoi']})
data3 = pd.DataFrame({'Educ': ['Aspirantura', 'Miena', 'Miena_mahsus', 'Miena_nopurra', 'Oli', 'Oli_nopurra']})

    
def issue_a_loan(FamilySize, Loan_Amount, Loan_Term,
                 Monthly_repayment_amount_according_to_schedule, Grace_period,
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage, Gross_profit,
                 Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience):
    # Преобразование business_experience в числовой формат
    business_experience = int(business_experience)

    # Преобразование всех переменных в числовой формат
    input_data = [
        FamilySize, Loan_Amount, Loan_Term,
        Monthly_repayment_amount_according_to_schedule, Grace_period,
        Capital, Asset, Days_overdue, Number_of_overdue,
        Lending_stage, Gross_profit,
        Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
        Region_code, Direction_of_activity, Currency_code, Pledge_code,
        business_experience
    ]

    # Объединение закодированных данных
    input_data.extend(Married_encoded.values.flatten())
    input_data.extend(Nationality_encoded.values.flatten())
    input_data.extend(Educ_encoded.values.flatten())
    
                     
    # Преобразуем в массив numpy и делаем предсказание
    input_array = np.array(input_data).reshape(1, -1)
    prediction1 = regression1.predict(input_array)
    prediction2 = regression2.predict(input_array)
    total_pred = (prediction1 + prediction2) / 3
    
    prediction2_1 = regression1.predict_proba(input_array)
    prediction2_2 = regression2.predict_proba(input_array)
    total_pred2 = (prediction2_1 + prediction2_2) / 3
    return total_pred, total_pred2


def Delays_days(FamilySize, Loan_Amount, Loan_Term,
                 Monthly_repayment_amount_according_to_schedule, Grace_period,
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage, Gross_profit,
                 Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience):
    # Исправлено: добавлено преобразование business_experience в числовой формат
    business_experience = int(business_experience)

    input_data = [
        FamilySize, Loan_Amount, Loan_Term,
        Monthly_repayment_amount_according_to_schedule, Grace_period,
        Capital, Asset, Days_overdue, Number_of_overdue,
        Lending_stage, Gross_profit,
        Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
        Region_code, Direction_of_activity, Currency_code, Pledge_code,
        business_experience
    ]

    # Объединение закодированных данных
    input_data = pd.concat([pd.Series(input_data), Married_encoded, Nationality_encoded, Educ_encoded], axis=0)

    # Преобразуем в массив numpy и делаем предсказание
    input_array = np.array(input_data).reshape(1, -1)
    reg1 = regression1.predict(input_array)
    return reg1  

def Credit_sum(FamilySize, Loan_Amount, Loan_Term,
                 Monthly_repayment_amount_according_to_schedule, Grace_period,
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage, Gross_profit,
                 Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience):
    # Исправлено: добавлено преобразование business_experience в числовой формат
    business_experience = int(business_experience)

    input_data = [
        FamilySize, Loan_Amount, Loan_Term,
        Monthly_repayment_amount_according_to_schedule, Grace_period,
        Capital, Asset, Days_overdue, Number_of_overdue,
        Lending_stage, Gross_profit,
        Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
        Region_code, Direction_of_activity, Currency_code, Pledge_code,
        business_experience
    ]

    # Объединение закодированных данных
    input_data = pd.concat([pd.Series(input_data), Married_encoded, Nationality_encoded, Educ_encoded], axis=0)

    # Преобразуем в массив numpy и делаем предсказание
    input_array = np.array(input_data).reshape(1, -1)
    reg2 = regression2.predict(input_array)    
    return reg2
                   
# Основная функция
def main():
    FamilySize = st.number_input('Сколько у вас членов семьи?', step=1, value=0)
    Loan_amount = st.number_input('На какую сумму хотите взять кредит?', step=1, value=0)
    Loan_term = st.number_input('На какой срок вы хотите взять кредит(месяц)?', step=1, value=0) 
    Monthly_repayment_amount_according_to_schedule = st.number_input('Ежемесячная сумма погашения?', step=1, value=0) 
    Grace_period = st.number_input('Льготный период?', step=1, value=0)
    Capital = st.number_input('Ваш начальный капитал?', step=1, value=0)
    Asset = st.number_input('Ваш актив?', step=1, value=0)
    Days_overdue = st.number_input('Дни просрочки?', step=1, value=0)
    Number_of_overdue = st.number_input('Дни количество дней просрочки?', step=1, value=0)
    Lending_stage = st.number_input('Какой раз вы уже получаете кредит?', step=1, value=0)
    Gross_profit = st.number_input('Валовая прибыль', step=1, value=0)
    Net_Profit = st.number_input('Чистая прибыль', step=1, value=0)
    Age = st.number_input('Сколько вам лет? запишите ваш год рождения в формате ', step=1, value=0)


    # Создаем dummy-столбцы со значениями False для всех категорий
    Married_encoded = pd.get_dummies(data1, prefix='', prefix_sep='').astype(bool)
    # Получаем выбор пользователя
    selected_Married = st.selectbox('Введите ваше семейное положение:', data1)
    # Если выбор сделан, обновляем соответствующий dummy-столбец в True
    if selected_Married is not None:
        selected_Married = Married_encoded[selected_Married].astype(bool)
    
    isFemale = st.radio("Укажите свой пол:", ['Мужской', 'Женский'])
    if sex == 'Мужской':
        Мужской = 0
    else:
        Женский = 1

# Создаем dummy-столбцы со значениями False для всех категорий
    nationality_encoded = pd.get_dummies(data2, prefix='', prefix_sep='').astype(bool)
    # Получаем выбор пользователя
    selected_nationality = st.selectbox('Введите национальность:', data2)
    # Если выбор сделан, обновляем соответствующий dummy-столбец в True
    if selected_nationality is not None:
        nationality_encoded = nationality_encoded[selected_nationality].astype(bool)


    educ =  ['Аспирантура', 'Оли', 'Миёнаи махсус', 'Олии нопурра', 'Миёна', 'Миёнаи нопурра']
    education = st.selectbox('Уровень образования:', educ)
    label_encoder = LabelEncoder()
    encoded_educ = label_encoder.fit_transform(educ)
    Education = encoded_educ[educ.index(education)]



    

    Filial_code = st.radio("Укажите филиал банка, в котром вы получаете кредит:", ['Istaravshan', 'Khujand', 'G_Rasulov', 'Dushanbe', 'Isfara', 'Panjakent'])
    if filial == 'Istaravshan':
        Istaravshan = 0
    elif filial == 'Khujand':
        Khujand = 1
    elif filial == 'G_Rasulov':
        G_Rasulov = 2
    elif filial == 'Dushanbe':
        Dushanbe = 3
    elif filial == 'Isfara':
        Isfara = 4
    else:
        Panjakent = 5
        
    Region_code = st.radio("Укажите регион, в котром вы получаете кредит:", ['Shahriston', 'Guli Surh', 'Khujand-Center', 'Spitamen', 'Shark', 'Marhamat', 'Dushanbe', 'Navkent',
                           'Kistacuz', 'Khujand-Panchshanbe', 'Buston', 'Isfara_filial', 'Rudaki', 'Asht', 'Kalininobod',
                           'Sino', 'Isfara', 'Hisor', 'Zafarobod', 'Nichoni', 'Vahdat', 'Mehnatobod', 'Uyas', 'G_Rasulov',
                           'Konibodom', 'Dusti', 'Niyozbek','Istaravshan', 'Rogun','Gonchi', 'Chashmasor', 'Nofaroch', 'Obodi',
                           'Karakchikum', 'Obburdon', 'Kurush', 'Voruh', 'Gulyakandoz', 'Nekfayz', 'Somgor', 'Punuk', 'Panjakent',
                           'Kulkand', 'Oppon', 'Fayzobod', 'Tursunzoda', 'Gusar', 'Ravshan','Iftihor', 'H_Aliev', 'Yori',
                           'Muchun', 'Sarazm'])
     if region == 'Shahriston'':
        Shahriston' = 0
    elif region == 'Guli Surh':
        Guli Surh = 1
    elif region == 'Khujand-Center' :
        Khujand-Center = 2
    elif region == 'Spitamen':
        Spitamen = 3
    elif region == 'Shark':
        Shark = 4
    elif region == 'Marhamat' :
        Marhamat = 5
    elif region == 'Dushanbe':
        Dushanbe = 6
     elif region == 'Navkent':
        Navkent = 7
    elif region == 'Kistacuz' :
        Kistacuz = 8
    elif region == 'Khujand-Panchshanbe':
        Khujand-Panchshanbe = 9
    elif region == 'Buston':
        Buston = 10
    elif region == 'Isfara_filial' :
        Isfara_filial = 11
    elif region == 'Rudaki':
        Rudaki = 12
    elif region == 'Asht':
        Asht = 13
    elif region == 'Kalininobod' :
        Kalininobod = 14
    elif region == 'Sino':
        Sino = 15
    elif region == 'Isfara':
        Isfara = 16
    elif region == 'Hisor' :
        Hisor = 17
    elif region == 'Zafarobod':
        Zafarobod = 18
    elif region == 'Nichoni':
        Nichoni = 19
    elif region == 'Vahdat' :
        Vahdat = 20
    elif region == 'Mehnatobod':
        Mehnatobod = 21
    elif region == 'Uyas':
        Uyas = 22
    elif region == 'G_Rasulov':
        G_Rasulov = 23
    elif region == 'Konibodom':
        Konibodom = 24
    elif region == 'Dusti':
        Dusti = 25
    elif region == 'Niyozbek':
        Niyozbek = 26
    elif region == 'Istaravshan':
        Istaravshan = 27
    elif region == 'Rogun':
        Rogun = 28
    elif region == 'Gonchi':
        Gonchi = 29
    elif region == 'Chashmasor':
        Chashmasor = 30
    elif region == 'Nofaroch':
        Nofaroch = 31
    elif region == 'Obodi':
        Obodi = 32
    elif region == 'Karakchikum':
        Karakchikum = 33
    elif region == 'Obburdon':
        Obburdon = 34
    elif region == 'Kurush':
        Kurush = 35
    elif region == 'Voruh':
        Voruh = 36
    elif region == 'Gulyakandoz':
        Gulyakandoz = 37
    elif region == 'Nekfayz':
        Nekfayz = 38
    elif region == 'Somgor':
        Somgor = 39
    elif region == 'Punuk':
        Punuk = 40
    elif region == 'Panjakent':
        Panjakent = 41
    elif region == 'Kulkand':
        Kulkand = 42
    elif region == 'Oppon':
        Oppon = 43
    elif region == 'Fayzobod':
        Fayzobod = 44
    elif region == 'Tursunzoda':
        Tursunzoda = 45
    elif region == 'Gusar':
        Gusar = 46
    elif region == 'Ravshan':
        Ravshan = 47
    elif region == 'Iftihor':
        Iftihor = 48
    elif region == 'H_Aliev':
        H_Aliev = 49
    elif region == 'Yori':
        Yori = 50
    elif region == 'Muchun':
        Muchun = 51
    else:
        Sarazm = 52
    




    Direction_of_activity = st.radio("Укажите ваше направление деятельности:", ['Animal_husbandry_and_milk_processing', 'Purchase_of_equipment',
                                     'House_renovation', 'trade', 'Agriculture', 'Purchasing_furniture',
                                     'Payment_for_treatment', 'Carrying_out_events', 'Payment_for_travel',
                                     'Services', 'Transport_conversion', 'Needs',
                                     'Paying_for_education', 'Production', 'Buying_an_apartment',
                                     'Consumed_other', 'Repair_of_place_of_business', 'Agriculture',
                                     'Everyone', 'Fruit_Drying', 'Commercial'
])
    if activity == 'Animal_husbandry_and_milk_processing':
        Animal_husbandry_and_milk_processing = 0
    elif activity == 'Purchase_of_equipment':
        Purchase_of_equipment = 1
    elif activity == 'House_renovation':
        House_renovation = 2
    elif activity == 'trade':
        trade = 3
    elif activity == 'Agriculture':
        Agriculture = 4
    elif activity == 'Purchasing_furniture':
        Purchasing_furniture = 5
    elif region == 'Payment_for_treatment':
        Payment_for_treatment = 6
    elif region == 'Carrying_out_events':
        Carrying_out_events = 7
    elif region == 'Payment_for_travel':
        Payment_for_travel = 8
    elif region == 'Services':
        Services = 9
    elif region == 'Transport_conversion':
        Transport_conversion = 10
    elif region == 'Needs':
        Needs = 11
    elif region == 'Paying_for_education':
        Paying_for_education = 12
    elif region == 'Production':
        Production = 13
    elif region == 'Buying_an_apartment':
        Buying_an_apartment = 14
    elif region == 'Consumed_other':
        Consumed_other = 15
    elif region == 'Repair_of_place_of_business':
        Repair_of_place_of_business = 16
    elif region == 'Agriculture':
        Agriculture = 17
    elif region == 'Everyone':
        Everyone = 18
    elif region == 'Fruit_Drying':
        Fruit_Drying = 19
    else:
        Commercial = 20
    
     Currency_code = st.radio("В какой валюте вы бы хотели получить кредит:", ['Dollar', 'Somoni', Rubl])
    if currency == 'Dollar':
        Dollar = 0
    elif currency == 'Somoni':
        Somoni = 1
    else:
        Rubl = 2


     Pledge_code = st.radio("В какой валюте вы бы хотели получить кредит:", ['Group', 'Category_1', 'Category_2', 'Category_3', 'Category_4'])
    if Pledge == 'Group':
        Group = 0
    elif Pledge == 'Category_1':
        Category_1 = 1
    elif Pledge == 'Category_2':
        Category_2 = 2
    elif Pledge == 'Category_3':
        Category_3 = 3
    else:
        Category_4 = 4

 
    
    
    
    educ =  ['Аспирантура', 'Оли', 'Миёнаи махсус', 'Олии нопурра', 'Миёна', 'Миёнаи нопурра']
    education = st.selectbox('Уровень образования:', educ)
    label_encoder = LabelEncoder()
    encoded_educ = label_encoder.fit_transform(educ)
    Education = encoded_educ[educ.index(education)]

    
    


        
    options = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-40', '40-50', '50+']
    bus_exp =  st.radio("Стаж работы:", options)
    label_encoder = LabelEncoder()
    encoded_busEx = label_encoder.fit_transform(options)
    business_experience = encoded_busEx[options.index(bus_exp)]

    Net_profit = st.number_input('Доход (в месяц):', step=1, value=0)
        
    result1 = ""
    result2 = ""
    result3 = ""
    result4 = ""
    if st.button("Predict"):
        result1, result2 = issue_a_loan(FamilySize, Loan_Amount, Loan_Term,
                 Monthly_repayment_amount_according_to_schedule, Grace_period,
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage, Gross_profit,
                 Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience)
        if result1 == 0:
            result3 = Credit_sum(FamilySize, Loan_Amount, Loan_Term,
                 Monthly_repayment_amount_according_to_schedule, Grace_period,
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage, Gross_profit,
                 Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience, result1)
            st.success(f'Сумма вам не доступна')
            st.success(f'Максимально доступная сумма: {result3}')
        else:
            st.success(f'Вероятность выдачи кредита {result1[0]*100:.2f}%')
            st.success(f'Вероятность возврата кредита вовремя: {result2[0][0] * 100:.2f}%')
            result4 = Delays_days(FamilySize, Loan_Amount, Loan_Term,
                 Monthly_repayment_amount_according_to_schedule, Grace_period,
                 Capital, Asset, Days_overdue, Number_of_overdue,
                 Lending_stage, Gross_profit,
                 Net_Profit, Age, Married_encoded, isFemale, nationality_encoded, educ, Filial_code,
                 Region_code, Direction_of_activity, Currency_code, Pledge_code,
                 business_experience)
            st.success(f'Примерная просрочка: {(result4[0]).astype(int)}')
                                                   
if __name__ == '__main__':
    main()
